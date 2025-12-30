//! Kelly Criterion Bet Sizing
//!
//! Optimal bet sizing based on edge and odds using Kelly criterion.
//!
//! The Kelly criterion formula:
//!     f* = (b*p - q) / b = (p*odds - 1) / (odds - 1)
//!
//! Where:
//!     f* = fraction of bankroll to bet
//!     b = odds - 1 (net odds)
//!     p = probability of winning
//!     q = 1 - p (probability of losing)
//!     odds = decimal odds (e.g., 5.0 means 5x return)

use serde::{Deserialize, Serialize};

/// Bet sizing recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BetSizing {
    pub probability: f64,
    pub odds: f64,
    pub expected_value: f64,
    pub edge: f64,                 // EV - 1
    pub kelly_fraction: f64,       // Full Kelly
    pub recommended_fraction: f64, // After applying Kelly multiplier
    pub stake: i64,                // Recommended stake amount
}

/// Calculate Kelly fraction for a single bet
///
/// # Arguments
/// * `probability` - Estimated probability of winning (0-1)
/// * `odds` - Decimal odds (e.g., 5.0 = 5x return)
///
/// # Returns
/// Kelly fraction (can be negative if EV < 1)
///
/// # Examples
/// ```
/// use boatrace::core::kelly::calculate_kelly_fraction;
/// let kelly = calculate_kelly_fraction(0.25, 5.0); // EV = 1.25
/// assert!((kelly - 0.0625).abs() < 0.0001);
/// ```
pub fn calculate_kelly_fraction(probability: f64, odds: f64) -> f64 {
    if odds <= 1.0 {
        return 0.0;
    }

    // f* = (p * odds - 1) / (odds - 1)
    (probability * odds - 1.0) / (odds - 1.0)
}

/// Calculate optimal stake amount
///
/// # Arguments
/// * `probability` - Estimated probability of winning
/// * `odds` - Decimal odds
/// * `bankroll` - Current bankroll amount
/// * `kelly_multiplier` - Kelly fraction multiplier (default: 0.25 = quarter Kelly)
/// * `min_stake` - Minimum stake amount (default: 100 yen)
/// * `max_stake_pct` - Maximum stake as percentage of bankroll (default: 10%)
///
/// # Returns
/// Recommended stake amount (rounded to 100 yen)
pub fn calculate_optimal_stake(
    probability: f64,
    odds: f64,
    bankroll: i64,
    kelly_multiplier: f64,
    min_stake: i64,
    max_stake_pct: f64,
) -> i64 {
    let kelly = calculate_kelly_fraction(probability, odds);

    if kelly <= 0.0 {
        return 0;
    }

    // Apply Kelly multiplier (fractional Kelly)
    let fraction = kelly * kelly_multiplier;

    // Calculate stake
    let mut stake = (bankroll as f64 * fraction) as i64;

    // Apply maximum stake limit
    let max_stake = (bankroll as f64 * max_stake_pct) as i64;
    stake = stake.min(max_stake);

    // Round to nearest 100 yen
    stake = (stake / 100) * 100;

    // Apply minimum stake
    if stake < min_stake {
        let raw_stake = (bankroll as f64 * kelly * kelly_multiplier) as i64;
        if raw_stake < min_stake / 2 {
            return 0;
        } else {
            return min_stake;
        }
    }

    stake
}

/// Kelly criterion calculator for bet sizing
///
/// Supports:
/// - Full Kelly (aggressive)
/// - Fractional Kelly (conservative, default 1/4)
/// - Multiple simultaneous bets
#[derive(Debug, Clone)]
pub struct KellyCalculator {
    pub bankroll: i64,
    pub kelly_multiplier: f64,
    pub min_stake: i64,
    pub max_stake_pct: f64,
    pub max_total_exposure: f64,
}

impl KellyCalculator {
    /// Create a new Kelly calculator
    ///
    /// # Arguments
    /// * `bankroll` - Initial bankroll
    /// * `kelly_multiplier` - Fraction of Kelly to use (0.25 = quarter Kelly)
    /// * `min_stake` - Minimum bet size
    /// * `max_stake_pct` - Maximum single bet as % of bankroll
    /// * `max_total_exposure` - Maximum total exposure across all bets
    pub fn new(
        bankroll: i64,
        kelly_multiplier: f64,
        min_stake: i64,
        max_stake_pct: f64,
        max_total_exposure: f64,
    ) -> Self {
        Self {
            bankroll,
            kelly_multiplier,
            min_stake,
            max_stake_pct,
            max_total_exposure,
        }
    }

    /// Create with default settings (100,000 bankroll, quarter Kelly)
    pub fn with_defaults(bankroll: i64) -> Self {
        Self {
            bankroll,
            kelly_multiplier: 0.25,
            min_stake: 100,
            max_stake_pct: 0.10,
            max_total_exposure: 0.30,
        }
    }

    /// Calculate bet sizing for a single bet
    pub fn calculate_single(&self, probability: f64, odds: f64) -> BetSizing {
        let ev = probability * odds;
        let edge = ev - 1.0;
        let kelly = calculate_kelly_fraction(probability, odds);
        let recommended = (kelly * self.kelly_multiplier).max(0.0);

        let stake = calculate_optimal_stake(
            probability,
            odds,
            self.bankroll,
            self.kelly_multiplier,
            self.min_stake,
            self.max_stake_pct,
        );

        BetSizing {
            probability,
            odds,
            expected_value: ev,
            edge,
            kelly_fraction: kelly,
            recommended_fraction: recommended,
            stake,
        }
    }

    /// Calculate bet sizing for multiple simultaneous bets
    ///
    /// When betting on multiple outcomes, we need to consider:
    /// 1. Total exposure limit
    /// 2. Overlapping events (same race = mutually exclusive)
    pub fn calculate_multiple(&self, bets: &[(f64, f64)]) -> Vec<BetSizing> {
        if bets.is_empty() {
            return Vec::new();
        }

        // Calculate individual sizing first
        let mut sizings: Vec<BetSizing> = bets
            .iter()
            .map(|(p, o)| self.calculate_single(*p, *o))
            .collect();

        // Calculate total requested stake
        let total_stake: i64 = sizings.iter().map(|s| s.stake).sum();

        // Check if we exceed max total exposure
        let max_exposure = (self.bankroll as f64 * self.max_total_exposure) as i64;

        if total_stake > max_exposure {
            // Scale down proportionally
            let scale_factor = max_exposure as f64 / total_stake as f64;

            for sizing in &mut sizings {
                let new_stake = ((sizing.stake as f64 * scale_factor) as i64 / 100) * 100;
                if new_stake < self.min_stake {
                    sizing.stake = 0;
                } else {
                    sizing.stake = new_stake;
                }
                sizing.recommended_fraction *= scale_factor;
            }
        }

        sizings
    }

    /// Update bankroll after bet result
    pub fn update_bankroll(&mut self, profit: i64) {
        self.bankroll += profit;
    }

    /// Simulate bankroll growth over a series of bets
    ///
    /// # Arguments
    /// * `bets` - List of (probability, odds, outcome) tuples
    ///
    /// # Returns
    /// List of bankroll values after each bet
    pub fn simulate_growth(&mut self, bets: &[(f64, f64, bool)]) -> Vec<i64> {
        let mut history = vec![self.bankroll];

        for (prob, odds, won) in bets {
            let sizing = self.calculate_single(*prob, *odds);

            if sizing.stake > 0 {
                let profit = if *won {
                    (sizing.stake as f64 * (odds - 1.0)) as i64
                } else {
                    -sizing.stake
                };

                self.update_bankroll(profit);
            }

            history.push(self.bankroll);
        }

        history
    }
}

impl Default for KellyCalculator {
    fn default() -> Self {
        Self::with_defaults(100_000)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kelly_fraction_positive_ev() {
        // EV = 0.25 * 5.0 = 1.25 (positive edge)
        let kelly = calculate_kelly_fraction(0.25, 5.0);
        assert!((kelly - 0.0625).abs() < 0.0001);
    }

    #[test]
    fn test_kelly_fraction_negative_ev() {
        // EV = 0.10 * 5.0 = 0.50 (negative edge)
        let kelly = calculate_kelly_fraction(0.10, 5.0);
        assert!(kelly < 0.0);
    }

    #[test]
    fn test_kelly_fraction_zero_odds() {
        let kelly = calculate_kelly_fraction(0.25, 1.0);
        assert_eq!(kelly, 0.0);
    }

    #[test]
    fn test_optimal_stake_calculation() {
        let stake = calculate_optimal_stake(0.25, 5.0, 100_000, 0.25, 100, 0.10);
        // Kelly = 0.0625, quarter Kelly = 0.0156, stake = 1560 -> 1500 (rounded)
        assert!(stake > 0);
        assert!(stake <= 10_000); // max 10%
    }

    #[test]
    fn test_calculator_single() {
        let calc = KellyCalculator::with_defaults(100_000);
        let sizing = calc.calculate_single(0.25, 5.0);

        assert!((sizing.expected_value - 1.25).abs() < 0.01);
        assert!((sizing.edge - 0.25).abs() < 0.01);
        assert!(sizing.stake > 0);
    }

    #[test]
    fn test_calculator_multiple_exposure_limit() {
        let calc = KellyCalculator::with_defaults(100_000);

        // Three high-value bets that would exceed 30% exposure
        let bets = vec![
            (0.25, 5.0), // EV = 1.25
            (0.20, 6.0), // EV = 1.20
            (0.15, 8.0), // EV = 1.20
        ];

        let sizings = calc.calculate_multiple(&bets);
        let total: i64 = sizings.iter().map(|s| s.stake).sum();

        // Total should not exceed 30% of bankroll
        assert!(total <= 30_000);
    }

    #[test]
    fn test_simulate_growth() {
        let mut calc = KellyCalculator::with_defaults(100_000);

        let bets = vec![
            (0.25, 5.0, true),  // Win
            (0.25, 5.0, false), // Lose
            (0.25, 5.0, true),  // Win
        ];

        let history = calc.simulate_growth(&bets);

        assert_eq!(history.len(), 4); // Initial + 3 bets
        assert_eq!(history[0], 100_000);
        assert!(history[1] > history[0]); // Won first bet
    }

    #[test]
    fn test_kelly_calculator_new() {
        let calc = KellyCalculator::new(
            50_000, // bankroll
            0.5,    // kelly_multiplier (half Kelly)
            200,    // min_stake
            0.15,   // max_stake_pct
            0.25,   // max_total_exposure
        );

        assert_eq!(calc.bankroll, 50_000);
        assert!((calc.kelly_multiplier - 0.5).abs() < 0.001);
        assert_eq!(calc.min_stake, 200);
        assert!((calc.max_stake_pct - 0.15).abs() < 0.001);
        assert!((calc.max_total_exposure - 0.25).abs() < 0.001);
    }

    #[test]
    fn test_update_bankroll() {
        let mut calc = KellyCalculator::with_defaults(100_000);

        // Win: +5000
        calc.update_bankroll(5000);
        assert_eq!(calc.bankroll, 105_000);

        // Lose: -3000
        calc.update_bankroll(-3000);
        assert_eq!(calc.bankroll, 102_000);
    }

    #[test]
    fn test_calculate_single_negative_ev() {
        let calc = KellyCalculator::with_defaults(100_000);

        // Negative EV bet (probability too low for odds)
        let sizing = calc.calculate_single(0.10, 5.0); // EV = 0.5

        assert!(sizing.expected_value < 1.0);
        assert!(sizing.kelly_fraction < 0.0);
        assert_eq!(sizing.stake, 0); // Should not bet
    }

    #[test]
    fn test_calculate_multiple_empty() {
        let calc = KellyCalculator::with_defaults(100_000);
        let sizings = calc.calculate_multiple(&[]);
        assert!(sizings.is_empty());
    }
}
