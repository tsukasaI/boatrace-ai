//! Synthetic Odds Generation
//!
//! Generate realistic market odds for backtesting when historical odds are unavailable.

use std::collections::HashMap;

/// Historical win rate by course position
const HISTORICAL_WIN_RATE: [f64; 6] = [0.55, 0.14, 0.12, 0.10, 0.06, 0.03];

/// Historical 2nd place rate by course position
const HISTORICAL_SECOND_RATE: [f64; 6] = [0.20, 0.22, 0.20, 0.18, 0.12, 0.08];

/// Synthetic odds generator
pub struct SyntheticOddsGenerator {
    /// Commission rate (takeout). 25% is standard for boat racing
    margin: f64,
    /// Pre-calculated exacta probabilities
    exacta_probs: HashMap<(u8, u8), f64>,
}

impl SyntheticOddsGenerator {
    /// Create a new synthetic odds generator
    ///
    /// # Arguments
    /// * `margin` - Commission rate (takeout). Default is 0.25 (25%)
    pub fn new(margin: f64) -> Self {
        let exacta_probs = Self::calculate_exacta_probs();
        Self {
            margin,
            exacta_probs,
        }
    }

    /// Calculate exacta probability for each course combination
    ///
    /// P(1st=i, 2nd=j) = P(1st=i) × P(2nd=j | 1st≠j)
    fn calculate_exacta_probs() -> HashMap<(u8, u8), f64> {
        let mut probs = HashMap::new();

        for first in 1u8..=6 {
            let p_first = HISTORICAL_WIN_RATE[(first - 1) as usize];

            // Calculate remaining 2nd place probability total
            let remaining_second_total: f64 = (1u8..=6)
                .filter(|&s| s != first)
                .map(|s| HISTORICAL_SECOND_RATE[(s - 1) as usize])
                .sum();

            for second in 1u8..=6 {
                if first == second {
                    continue;
                }

                // Calculate 2nd place using conditional probability
                let p_second_given =
                    HISTORICAL_SECOND_RATE[(second - 1) as usize] / remaining_second_total;
                let p_exacta = p_first * p_second_given;

                probs.insert((first, second), p_exacta);
            }
        }

        // Normalize so total equals 1
        let total: f64 = probs.values().sum();
        for prob in probs.values_mut() {
            *prob /= total;
        }

        probs
    }

    /// Get odds for specified combination
    ///
    /// # Arguments
    /// * `first` - Boat number for 1st place (1-6)
    /// * `second` - Boat number for 2nd place (1-6)
    ///
    /// # Returns
    /// Odds (payout multiplier)
    pub fn get_odds(&self, first: u8, second: u8) -> f64 {
        if first == second || first < 1 || first > 6 || second < 1 || second > 6 {
            return 0.0;
        }

        let prob = *self.exacta_probs.get(&(first, second)).unwrap_or(&0.001);

        // Odds = 1 / (probability × (1 - commission rate))
        let fair_odds = 1.0 / prob;
        let actual_odds = fair_odds * (1.0 - self.margin);

        (actual_odds * 10.0).round() / 10.0 // Round to 1 decimal
    }

    /// Get odds for all 30 combinations
    ///
    /// # Returns
    /// HashMap of {(first, second): odds}
    pub fn get_all_odds(&self) -> HashMap<(u8, u8), f64> {
        let mut odds = HashMap::new();
        for first in 1u8..=6 {
            for second in 1u8..=6 {
                if first != second {
                    odds.insert((first, second), self.get_odds(first, second));
                }
            }
        }
        odds
    }

    /// Get the pre-calculated exacta probabilities
    pub fn get_exacta_probs(&self) -> &HashMap<(u8, u8), f64> {
        &self.exacta_probs
    }
}

impl Default for SyntheticOddsGenerator {
    fn default() -> Self {
        Self::new(0.25) // 25% margin is standard
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synthetic_odds_generator_new() {
        let generator = SyntheticOddsGenerator::new(0.25);
        assert!((generator.margin - 0.25).abs() < 0.001);
        assert_eq!(generator.exacta_probs.len(), 30); // 6 × 5 combinations
    }

    #[test]
    fn test_exacta_probs_sum_to_one() {
        let generator = SyntheticOddsGenerator::default();
        let total: f64 = generator.exacta_probs.values().sum();
        assert!(
            (total - 1.0).abs() < 0.001,
            "Total probability {} should be ~1.0",
            total
        );
    }

    #[test]
    fn test_get_odds_valid() {
        let generator = SyntheticOddsGenerator::default();

        // 1-2 should have lowest odds (most likely)
        let odds_1_2 = generator.get_odds(1, 2);
        assert!(odds_1_2 > 1.0, "Odds should be > 1.0");
        assert!(odds_1_2 < 10.0, "1-2 should have low odds");

        // 6-5 should have higher odds (less likely)
        let odds_6_5 = generator.get_odds(6, 5);
        assert!(odds_6_5 > odds_1_2, "6-5 should have higher odds than 1-2");
    }

    #[test]
    fn test_get_odds_invalid() {
        let generator = SyntheticOddsGenerator::default();

        assert_eq!(generator.get_odds(1, 1), 0.0); // Same boat
        assert_eq!(generator.get_odds(0, 2), 0.0); // Invalid first
        assert_eq!(generator.get_odds(1, 7), 0.0); // Invalid second
    }

    #[test]
    fn test_get_all_odds() {
        let generator = SyntheticOddsGenerator::default();
        let all_odds = generator.get_all_odds();

        assert_eq!(all_odds.len(), 30);

        // All odds should be positive
        for &odds in all_odds.values() {
            assert!(odds > 0.0);
        }
    }

    #[test]
    fn test_margin_effect() {
        let gen_low = SyntheticOddsGenerator::new(0.10); // 10% margin
        let gen_high = SyntheticOddsGenerator::new(0.30); // 30% margin

        // Higher margin = lower odds
        let odds_low = gen_low.get_odds(1, 2);
        let odds_high = gen_high.get_odds(1, 2);

        assert!(odds_low > odds_high, "Higher margin should result in lower odds");
    }

    #[test]
    fn test_course_1_advantage() {
        let generator = SyntheticOddsGenerator::default();

        // Combinations starting with course 1 should have lower odds
        let avg_course_1: f64 = (2u8..=6)
            .map(|s| generator.get_odds(1, s))
            .sum::<f64>()
            / 5.0;

        let avg_course_6: f64 = (1u8..=5)
            .map(|s| generator.get_odds(6, s))
            .sum::<f64>()
            / 5.0;

        assert!(
            avg_course_1 < avg_course_6,
            "Course 1 combinations should have lower avg odds"
        );
    }
}
