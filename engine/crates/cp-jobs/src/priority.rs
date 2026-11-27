//! Job classes and priorities.

/// Coarse urgency of a job.
///
/// A queued job of a more urgent class always starts before one of a less urgent class,
/// whatever their keys. `Ord` follows urgency: `Critical < Interactive < Streaming < Background`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum JobClass {
    /// Work the running frame or tick is blocked on.
    Critical = 0,
    /// Work the player is waiting for, such as remeshing an edited chunk.
    Interactive = 1,
    /// Generation and meshing for world streaming.
    Streaming = 2,
    /// Work without a deadline.
    Background = 3,
}

impl JobClass {
    /// Number of classes.
    pub const COUNT: usize = 4;

    /// Every class, most urgent first.
    pub const ALL: [Self; Self::COUNT] = [
        Self::Critical,
        Self::Interactive,
        Self::Streaming,
        Self::Background,
    ];

    /// The class as an array index in `0..COUNT`; equals its discriminant.
    #[must_use]
    pub const fn index(self) -> usize {
        self as usize
    }

    /// Whether this is a *bulk* class: [`Streaming`](Self::Streaming) or
    /// [`Background`](Self::Background), work that comes in large amounts and that no frame or
    /// tick waits for.
    ///
    /// A job system may limit how many bulk jobs run at the same time
    /// ([`WorkerPolicy::bulk_limit`](crate::WorkerPolicy::bulk_limit)). Jobs of the other
    /// classes are never held back by that limit.
    ///
    /// The bulk classes are the least urgent ones: each of them sorts after each other class.
    /// The queue relies on that.
    #[must_use]
    pub const fn is_bulk(self) -> bool {
        matches!(self, Self::Streaming | Self::Background)
    }
}

/// When a job starts relative to the other queued jobs.
///
/// Jobs start in ascending order of `class`, then ascending `key`; jobs with the same class and
/// key start in the order of their `spawn` calls (calls that overlap in time on different
/// threads have no order to keep). `Ord` follows the same rule: the smaller priority starts
/// first.
///
/// Priorities order the *start* of jobs only. A running job is never preempted.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Priority {
    /// Urgency class; compared first.
    pub class: JobClass,
    /// Order within the class: lower keys start first. The unit is up to the caller, for
    /// example a quantised distance to the camera.
    pub key: u32,
}

impl Priority {
    /// A priority of `class` with `key`.
    #[must_use]
    pub const fn new(class: JobClass, key: u32) -> Self {
        Self { class, key }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classes_are_ordered_by_urgency_and_indexed_by_discriminant() {
        assert!(JobClass::ALL.is_sorted());
        for (position, class) in JobClass::ALL.into_iter().enumerate() {
            assert_eq!(class.index(), position);
            assert_eq!(class as u8 as usize, position);
        }
        assert_eq!(JobClass::ALL.len(), JobClass::COUNT);
    }

    #[test]
    fn bulk_classes_are_streaming_and_background_and_sort_last() {
        assert_eq!(
            JobClass::ALL.map(JobClass::is_bulk),
            [false, false, true, true]
        );
        // The queue takes the first queued job or none: behind a bulk job there must be bulk
        // jobs only.
        for urgent in JobClass::ALL.into_iter().filter(|class| !class.is_bulk()) {
            for bulk in JobClass::ALL.into_iter().filter(|class| class.is_bulk()) {
                assert!(urgent < bulk, "{urgent:?} must sort before {bulk:?}");
                assert!(
                    Priority::new(urgent, u32::MAX) < Priority::new(bulk, 0),
                    "{urgent:?} must start before {bulk:?} whatever the keys"
                );
            }
        }
    }

    #[test]
    fn class_is_compared_before_key() {
        let urgent_late = Priority::new(JobClass::Critical, u32::MAX);
        let relaxed_early = Priority::new(JobClass::Background, 0);
        assert!(urgent_late < relaxed_early);

        let near = Priority::new(JobClass::Streaming, 3);
        let far = Priority::new(JobClass::Streaming, 4);
        assert!(near < far);
        assert_eq!(
            near,
            Priority {
                class: JobClass::Streaming,
                key: 3
            }
        );
    }
}
