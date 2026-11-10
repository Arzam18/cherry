use std::{sync::atomic::*, time::*};

use crate::*;

/*----------------------------------------------------------------*/

pub const DEFAULT_OVERHEAD: u64 = 100;

// Poll the hard deadline frequently enough that a slow mobile CPU cannot
// overshoot it by a large amount. BatchedAtomicCounter exposes its pending
// buffer through local(), so this does not depend on flush boundaries.
const TIME_CHECK_INTERVAL_NODES: u64 = 128;
const HARD_TIME_SOFT_MULTIPLIER: u64 = 4;
const MAX_HARD_TIME_CLOCK_DIVISOR: u64 = 4;
const SOFT_TARGET_MOVE_TIME_GRACE_MS: u64 = 250;

#[inline]
fn movetime_hard_budget(time: u64) -> u64 {
    time.saturating_add((time / 4).min(SOFT_TARGET_MOVE_TIME_GRACE_MS))
}

/// Calculate the initial soft and hard budgets for a clock-controlled search.
/// Cherry keeps its tuned, increment-aware allocation and later adjusts the
/// soft target using move/score stability. The hard deadline is deliberately
/// stricter: no more than four times the initial soft target and no more than
/// 25% of the usable remaining clock. The four-times ratio follows the
/// soft/hard separation used by Rodent V (75% soft, 3x allocation hard), while
/// the 25% reserve follows Pea's conservative mobile-clock ceiling.
#[inline]
fn clock_budgets(time: u64, inc: u64, moves_to_go: Option<u16>) -> (u64, u64) {
    let (soft_time, tuned_hard_time) = if let Some(moves_to_go) = moves_to_go.filter(|n| *n > 0) {
        let moves_to_go = u64::from(moves_to_go);
        let hard_time = ((time as f64 / (W::hard_time_div() as f64 / 4096.0)) as u64)
            .saturating_add(inc)
            .min(time);
        let soft_time = (time / moves_to_go).saturating_add(inc).min(hard_time);
        (soft_time, hard_time)
    } else {
        let hard_time = ((time as f64 / (W::hard_time_div() as f64 / 4096.0)) as u64)
            .saturating_add(inc.saturating_mul(W::hard_time_inc()) / 4096)
            .min(time);
        let soft_time = ((time as f64 / (W::soft_time_div() as f64 / 4096.0)) as u64)
            .saturating_add(inc.saturating_mul(W::soft_time_inc()) / 4096)
            .min(hard_time);
        (soft_time, hard_time)
    };

    // Do not spend the increment before it is actually received. In
    // particular, a large increment with a low remaining clock must not turn
    // the hard deadline into the entire current clock.
    let safe_clock = time / MAX_HARD_TIME_CLOCK_DIVISOR;
    let soft_time = soft_time.min(safe_clock);
    let hard_cap = soft_time
        .saturating_mul(HARD_TIME_SOFT_MULTIPLIER)
        .min(safe_clock);
    let hard_time = tuned_hard_time.min(hard_cap).min(time);
    (soft_time.min(hard_time), hard_time)
}

#[derive(Debug, Clone)]
pub enum SearchLimit {
    SearchMoves(MoveList),
    WhiteTime(u64),
    BlackTime(u64),
    WhiteInc(u64),
    BlackInc(u64),
    MoveTime(u64),
    MovesToGo(u16),
    Nodes(u64),
    Depth(u8),
}

/*----------------------------------------------------------------*/

pub struct TimeManager {
    start: AtomicInstant,
    infinite: AtomicBool,
    check_time: AtomicBool,
    no_manage: AtomicBool,
    stop: AtomicU32,

    base_time: AtomicU64,
    soft_time: AtomicU64,
    hard_time: AtomicU64,

    soft_nodes: AtomicU64,
    hard_nodes: AtomicU64,
    depth: AtomicU8,
}

impl TimeManager {
    #[inline]
    pub fn new() -> Self {
        TimeManager {
            start: AtomicInstant::now(),
            infinite: AtomicBool::new(true),
            check_time: AtomicBool::new(false),
            no_manage: AtomicBool::new(false),
            stop: AtomicU32::new(0),

            base_time: AtomicU64::new(0),
            soft_time: AtomicU64::new(0),
            hard_time: AtomicU64::new(0),

            soft_nodes: AtomicU64::new(u64::MAX),
            hard_nodes: AtomicU64::new(u64::MAX),
            depth: AtomicU8::new(MAX_DEPTH),
        }
    }

    #[inline]
    pub fn init(&self, stm: Color, limits: &[SearchLimit], overhead: u64, soft_target: bool) {
        self.set_stop(false);

        let mut inc = [0; Color::COUNT];
        let mut time = [u64::MAX; Color::COUNT];
        let mut moves_to_go = None;
        let mut move_time = None;
        let mut nodes = u64::MAX;
        let mut depth = MAX_DEPTH;
        let mut infinite = true;
        let mut check_time = false;
        let mut no_manage = false;

        for limit in limits {
            use SearchLimit::*;

            match *limit {
                SearchMoves(_) => {}
                WhiteTime(t) => time[Color::White] = t,
                BlackTime(t) => time[Color::Black] = t,
                WhiteInc(i) => inc[Color::White] = i,
                BlackInc(i) => inc[Color::Black] = i,
                MoveTime(t) => move_time = Some(t),
                MovesToGo(n) => moves_to_go = Some(n),
                Nodes(n) => nodes = nodes.min(n),
                Depth(d) => depth = depth.min(d),
            }

            if matches!(
                limit,
                WhiteTime(..) | BlackTime(..) | MoveTime(..) | Depth(..) | Nodes(..)
            ) {
                infinite = false;
            }

            if matches!(limit, WhiteTime(..) | BlackTime(..) | MoveTime(..)) {
                check_time = true;
            }

            if matches!(limit, MoveTime(..)) {
                no_manage = true;
            }
        }

        self.infinite.store(infinite, Ordering::Relaxed);
        self.check_time.store(check_time, Ordering::Relaxed);
        self.no_manage.store(no_manage, Ordering::Relaxed);

        self.depth.store(depth, Ordering::Relaxed);
        self.soft_nodes.store(nodes, Ordering::Relaxed);
        if soft_target {
            self.hard_nodes
                .store(nodes.saturating_mul(2000), Ordering::Relaxed);
        } else {
            self.hard_nodes.store(nodes, Ordering::Relaxed);
        }

        if let Some(time) = move_time {
            self.base_time.store(time, Ordering::Relaxed);
            self.soft_time.store(time, Ordering::Relaxed);

            if soft_target {
                self.hard_time
                    .store(movetime_hard_budget(time), Ordering::Relaxed);
            } else {
                self.hard_time.store(time, Ordering::Relaxed);
            }
        } else {
            let (time, inc) = (time[stm].saturating_sub(overhead), inc[stm]);
            let (soft_time, hard_time) = clock_budgets(time, inc, moves_to_go);

            self.base_time.store(soft_time, Ordering::Relaxed);
            self.soft_time.store(soft_time, Ordering::Relaxed);
            self.hard_time.store(hard_time, Ordering::Relaxed);
        }

        self.start.store(Instant::now(), Ordering::Relaxed);
    }

    #[inline]
    pub fn deepen(
        &self,
        depth: u8,
        _score: Score,
        _static_eval: Score,
        move_stability: u8,
        score_stability: u8,
        move_nodes: u64,
        nodes: u64,
    ) {
        if depth < 4 || nodes == 0 || self.no_manage.load(Ordering::Relaxed) {
            return;
        }

        let move_stability_factor = (W::move_stability_base()
            - W::move_stability_scale() * move_stability as i64)
            .max(W::move_stability_min());
        let score_stability_factor = (W::score_stability_base()
            - W::score_stability_scale() * score_stability as i64)
            .max(W::score_stability_min());
        let subtree_factor = (W::subtree_base()
            - (W::subtree_scale() as f64 * move_nodes as f64 / nodes as f64) as i64)
            .max(W::subtree_min());

        let base_time = self.base_time.load(Ordering::Relaxed);
        let hard_time = self.hard_time.load(Ordering::Relaxed);
        let new_target = ((base_time as u128
            * move_stability_factor as u128
            * score_stability_factor as u128
            * subtree_factor as u128)
            / 4096u128.pow(3)) as u64;

        self.soft_time
            .store(new_target.min(hard_time), Ordering::Relaxed);
    }

    #[inline]
    pub fn set_stop(&self, stop: bool) {
        self.stop.store(stop as u32, Ordering::Relaxed);
        if self.infinite() {
            atomic_wait::wake_all(&self.stop);
        }
    }

    #[inline]
    pub fn wait_for_stop(&self) {
        while !self.should_stop() {
            atomic_wait::wait(&self.stop, 0);
        }
    }

    #[inline]
    pub fn stop_search(&self, thread: &ThreadData) -> bool {
        self.should_stop()
            || thread.nodes.global() >= self.hard_nodes.load(Ordering::Relaxed)
            || (thread.nodes.local().is_multiple_of(TIME_CHECK_INTERVAL_NODES)
                && thread.id == 0
                && self.check_time.load(Ordering::Relaxed)
                && self.elapsed().as_millis() as u64 >= self.hard_time.load(Ordering::Relaxed))
    }

    #[inline]
    pub fn stop_id(&self, depth: u8, nodes: u64) -> bool {
        self.should_stop()
            || depth >= self.depth.load(Ordering::Relaxed)
            || nodes >= self.soft_nodes.load(Ordering::Relaxed)
            || (self.check_time.load(Ordering::Relaxed)
                && self.elapsed().as_millis() as u64 >= self.soft_time.load(Ordering::Relaxed))
    }

    #[inline]
    pub fn elapsed(&self) -> Duration {
        self.start.load(Ordering::Relaxed).elapsed()
    }

    #[inline]
    pub fn should_stop(&self) -> bool {
        self.stop.load(Ordering::Relaxed) != 0
    }

    #[inline]
    pub fn infinite(&self) -> bool {
        self.infinite.load(Ordering::Relaxed)
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hard_budget_is_bounded_by_clock_reserve() {
        let (soft, hard) = clock_budgets(60_000, 0, None);
        assert!(soft > 0);
        assert!(hard >= soft);
        assert!(hard <= soft.saturating_mul(HARD_TIME_SOFT_MULTIPLIER));
        assert!(hard <= 60_000 / MAX_HARD_TIME_CLOCK_DIVISOR);
        assert!(hard <= 60_000);
    }

    #[test]
    fn zero_moves_to_go_uses_incremental_clock_formula() {
        assert_eq!(clock_budgets(30_000, 0, Some(0)), clock_budgets(30_000, 0, None));
    }

    #[test]
    fn increment_budget_never_exceeds_remaining_time() {
        let (soft, hard) = clock_budgets(5_000, 2_000, None);
        assert!(soft <= 5_000);
        assert!(hard <= 5_000);
        assert!(soft <= hard);
        assert!(hard <= 5_000 / MAX_HARD_TIME_CLOCK_DIVISOR);
        assert!(hard <= soft.saturating_mul(HARD_TIME_SOFT_MULTIPLIER));
    }

    #[test]
    fn movetime_soft_target_has_bounded_grace() {
        assert_eq!(movetime_hard_budget(1_000), 1_250);
        assert_eq!(movetime_hard_budget(100), 125);
        assert_eq!(movetime_hard_budget(u64::MAX), u64::MAX);
    }

    #[test]
    fn large_increment_cannot_consume_the_current_clock() {
        let (soft, hard) = clock_budgets(1_000, 1_000, None);
        assert!(soft <= 250);
        assert!(hard <= 250);
        assert!(soft <= hard);
    }

    #[test]
    fn moves_to_go_budget_respects_both_hard_caps() {
        let (soft, hard) = clock_budgets(60_000, 0, Some(25));
        assert!(soft > 0);
        assert!(soft <= 15_000);
        assert!(hard >= soft);
        assert!(hard <= soft.saturating_mul(HARD_TIME_SOFT_MULTIPLIER));
        assert!(hard <= 15_000);
    }

    #[test]
    fn no_usable_time_produces_zero_budget() {
        assert_eq!(clock_budgets(0, 1_000, None), (0, 0));
    }
}
