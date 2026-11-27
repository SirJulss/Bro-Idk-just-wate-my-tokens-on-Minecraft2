//! Which adapter the engine uses when the system offers several.
//!
//! wgpu's own `request_adapter` ranks by device type first and leaves the backend order to an
//! implementation detail. The contract wants "Vulkan, then DX12" with DX12 as the fallback, so
//! both the ranking and the fallback are implemented here, over plain data and testable
//! without a GPU.

use super::options::backend_rank;

/// The facts about an adapter that decide its rank.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Candidate {
    pub backend: wgpu::Backend,
    pub device_type: wgpu::DeviceType,
}

/// The indices of `candidates`, best adapter first.
///
/// Order of the criteria, most important first:
/// 1. hardware before software rasterisers - a real GPU on DX12 beats a CPU Vulkan driver;
/// 2. backend order (Vulkan, then DX12);
/// 3. the power preference within a backend;
/// 4. the driver's enumeration order.
pub(crate) fn rank_adapters(candidates: &[Candidate], power: wgpu::PowerPreference) -> Vec<usize> {
    let mut order: Vec<usize> = (0..candidates.len()).collect();
    order.sort_by_key(|&index| {
        let candidate = candidates[index];
        (
            candidate.device_type == wgpu::DeviceType::Cpu,
            backend_rank(candidate.backend),
            power_rank(candidate.device_type, power),
            index,
        )
    });
    order
}

/// Rank of a device type under a power preference; lower is preferred.
///
/// "Other" means "unknown" for most drivers, so it ranks behind the types that name real
/// hardware. Without a preference all adapters rank equal and the enumeration order decides.
fn power_rank(device_type: wgpu::DeviceType, power: wgpu::PowerPreference) -> u8 {
    use wgpu::DeviceType as Type;
    match power {
        wgpu::PowerPreference::None => 0,
        wgpu::PowerPreference::HighPerformance => match device_type {
            Type::DiscreteGpu => 0,
            Type::IntegratedGpu => 1,
            Type::Other => 2,
            Type::VirtualGpu => 3,
            Type::Cpu => 4,
        },
        wgpu::PowerPreference::LowPower => match device_type {
            Type::IntegratedGpu => 0,
            Type::DiscreteGpu => 1,
            Type::Other => 2,
            Type::VirtualGpu => 3,
            Type::Cpu => 4,
        },
    }
}

/// Tries the candidates in `order` until `attempt` succeeds for one of them.
///
/// Returns the first success. When every attempt fails, the error of the first candidate is
/// returned, because that is the adapter the engine wanted; `None` means `order` was empty.
/// This is what makes DX12 a fallback not only for a missing Vulkan driver but also for one
/// that enumerates an adapter and then fails to open it.
pub(crate) async fn first_success<T, E>(
    order: &[usize],
    mut attempt: impl AsyncFnMut(usize) -> Result<T, E>,
) -> Option<Result<T, E>> {
    let mut first_error = None;
    for &index in order {
        match attempt(index).await {
            Ok(value) => return Some(Ok(value)),
            Err(error) => {
                first_error.get_or_insert(error);
            }
        }
    }
    first_error.map(Err)
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_lite::future::block_on;
    use wgpu::{Backend, DeviceType, PowerPreference};

    const ALL_PREFERENCES: [PowerPreference; 3] = [
        PowerPreference::None,
        PowerPreference::LowPower,
        PowerPreference::HighPerformance,
    ];

    fn candidate(backend: Backend, device_type: DeviceType) -> Candidate {
        Candidate {
            backend,
            device_type,
        }
    }

    /// The preferred adapter.
    fn best(candidates: &[Candidate], power: PowerPreference) -> Option<usize> {
        rank_adapters(candidates, power).first().copied()
    }

    #[test]
    fn nothing_to_choose_from() {
        assert_eq!(best(&[], PowerPreference::HighPerformance), None);
    }

    #[test]
    fn ranking_is_a_permutation_of_all_candidates() {
        let candidates = [
            candidate(Backend::Dx12, DeviceType::Cpu),
            candidate(Backend::Dx12, DeviceType::DiscreteGpu),
            candidate(Backend::Vulkan, DeviceType::IntegratedGpu),
            candidate(Backend::Vulkan, DeviceType::Cpu),
            candidate(Backend::Vulkan, DeviceType::DiscreteGpu),
            candidate(Backend::Dx12, DeviceType::IntegratedGpu),
        ];
        for power in ALL_PREFERENCES {
            let mut order = rank_adapters(&candidates, power);
            order.sort_unstable();
            assert_eq!(order, [0, 1, 2, 3, 4, 5], "{power:?}");
        }
        // Vulkan hardware, then DX12 hardware, then software in backend order.
        assert_eq!(
            rank_adapters(&candidates, PowerPreference::HighPerformance),
            [4, 2, 1, 5, 3, 0]
        );
        assert_eq!(
            rank_adapters(&candidates, PowerPreference::LowPower),
            [2, 4, 5, 1, 3, 0]
        );
    }

    #[test]
    fn vulkan_wins_over_dx12_for_the_same_gpu() {
        let candidates = [
            candidate(Backend::Dx12, DeviceType::DiscreteGpu),
            candidate(Backend::Vulkan, DeviceType::DiscreteGpu),
        ];
        for power in ALL_PREFERENCES {
            assert_eq!(rank_adapters(&candidates, power), [1, 0], "{power:?}");
        }
    }

    #[test]
    fn backend_order_outranks_the_power_preference() {
        let candidates = [
            candidate(Backend::Dx12, DeviceType::DiscreteGpu),
            candidate(Backend::Vulkan, DeviceType::IntegratedGpu),
        ];
        assert_eq!(best(&candidates, PowerPreference::HighPerformance), Some(1));
    }

    #[test]
    fn dx12_is_the_fallback_when_vulkan_has_no_adapter() {
        let candidates = [
            candidate(Backend::Dx12, DeviceType::IntegratedGpu),
            candidate(Backend::Dx12, DeviceType::DiscreteGpu),
            candidate(Backend::Dx12, DeviceType::Cpu),
        ];
        assert_eq!(best(&candidates, PowerPreference::HighPerformance), Some(1));
        assert_eq!(best(&candidates, PowerPreference::LowPower), Some(0));
    }

    #[test]
    fn hardware_on_dx12_beats_software_on_vulkan() {
        let candidates = [
            candidate(Backend::Vulkan, DeviceType::Cpu),
            candidate(Backend::Dx12, DeviceType::IntegratedGpu),
        ];
        for power in ALL_PREFERENCES {
            assert_eq!(rank_adapters(&candidates, power), [1, 0], "{power:?}");
        }
    }

    #[test]
    fn software_is_used_when_nothing_else_exists() {
        let candidates = [
            candidate(Backend::Dx12, DeviceType::Cpu),
            candidate(Backend::Vulkan, DeviceType::Cpu),
        ];
        assert_eq!(best(&candidates, PowerPreference::HighPerformance), Some(1));
    }

    #[test]
    fn power_preference_picks_within_a_backend() {
        let candidates = [
            candidate(Backend::Vulkan, DeviceType::IntegratedGpu),
            candidate(Backend::Vulkan, DeviceType::DiscreteGpu),
            candidate(Backend::Vulkan, DeviceType::VirtualGpu),
            candidate(Backend::Vulkan, DeviceType::Other),
        ];
        assert_eq!(
            rank_adapters(&candidates, PowerPreference::HighPerformance),
            [1, 0, 3, 2]
        );
        assert_eq!(
            rank_adapters(&candidates, PowerPreference::LowPower),
            [0, 1, 3, 2]
        );
    }

    #[test]
    fn without_a_preference_the_enumeration_order_decides() {
        let candidates = [
            candidate(Backend::Vulkan, DeviceType::IntegratedGpu),
            candidate(Backend::Vulkan, DeviceType::DiscreteGpu),
        ];
        assert_eq!(rank_adapters(&candidates, PowerPreference::None), [0, 1]);
        let reversed = [candidates[1], candidates[0]];
        assert_eq!(rank_adapters(&reversed, PowerPreference::None), [0, 1]);
    }

    #[test]
    fn equal_candidates_keep_the_enumeration_order() {
        let candidates = [candidate(Backend::Vulkan, DeviceType::DiscreteGpu); 3];
        assert_eq!(
            rank_adapters(&candidates, PowerPreference::HighPerformance),
            [0, 1, 2]
        );
    }

    #[test]
    fn the_first_adapter_that_opens_is_used() {
        let mut tried = Vec::new();
        let outcome = block_on(first_success(&[4, 2, 7], async |index| {
            tried.push(index);
            if index == 2 { Ok("device") } else { Err(index) }
        }));
        assert_eq!(outcome, Some(Ok("device")));
        // The preferred adapter was tried first, and nothing after the success.
        assert_eq!(tried, [4, 2]);
    }

    #[test]
    fn the_preferred_adapter_is_used_without_trying_others() {
        let mut tried = Vec::new();
        let outcome: Option<Result<usize, ()>> = block_on(first_success(&[3, 1], async |index| {
            tried.push(index);
            Ok(index)
        }));
        assert_eq!(outcome, Some(Ok(3)));
        assert_eq!(tried, [3]);
    }

    #[test]
    fn when_every_adapter_fails_the_preferred_one_explains_why() {
        let mut tried = Vec::new();
        let outcome: Option<Result<(), String>> =
            block_on(first_success(&[5, 0, 9], async |index| {
                tried.push(index);
                Err(format!("adapter {index} failed"))
            }));
        assert_eq!(outcome, Some(Err("adapter 5 failed".to_owned())));
        assert_eq!(tried, [5, 0, 9]);
    }

    #[test]
    fn without_candidates_nothing_is_attempted() {
        let outcome: Option<Result<(), ()>> = block_on(first_success(&[], async |_| {
            panic!("there is no adapter to try");
        }));
        assert_eq!(outcome, None);
    }
}
