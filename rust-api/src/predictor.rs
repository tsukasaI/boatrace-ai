use crate::models::{RacerEntry, PositionProb, ExactaPrediction};

/// Predictor for boat race outcomes
///
/// TODO: Replace with ONNX model loading for production
pub struct Predictor {
    // Course advantage weights (boat 1 has ~55% win rate historically)
    course_advantage: [f64; 6],
}

impl Predictor {
    pub fn new() -> Self {
        Self {
            // Historical course advantage percentages
            course_advantage: [0.55, 0.14, 0.12, 0.10, 0.06, 0.03],
        }
    }

    /// Predict position probabilities for each boat
    ///
    /// Returns 6 boats × 6 positions probability matrix
    pub fn predict_positions(&self, entries: &[RacerEntry]) -> Vec<PositionProb> {
        let mut probs = Vec::with_capacity(6);

        // Calculate base scores from features
        let scores: Vec<f64> = entries.iter().map(|e| self.calculate_score(e)).collect();
        let total_score: f64 = scores.iter().sum();

        for (i, entry) in entries.iter().enumerate() {
            // Normalize score as win probability
            let base_prob = if total_score > 0.0 {
                scores[i] / total_score
            } else {
                1.0 / 6.0
            };

            // Blend with course advantage
            let course_idx = (entry.boat_no - 1) as usize;
            let win_prob = 0.6 * base_prob + 0.4 * self.course_advantage[course_idx];

            // Generate position distribution
            // Decreasing probability for worse positions
            let position_probs = self.generate_position_dist(win_prob);

            probs.push(PositionProb {
                boat_no: entry.boat_no,
                probs: position_probs,
            });
        }

        // Normalize across boats for each position
        self.normalize_probs(&mut probs);

        probs
    }

    /// Calculate a score for a racer based on features
    fn calculate_score(&self, entry: &RacerEntry) -> f64 {
        let mut score = 0.0;

        // Win rate (most important)
        score += entry.national_win_rate * 0.3;
        score += entry.local_win_rate * 0.2;

        // In-2 rate
        score += entry.national_in2_rate * 0.01;
        score += entry.local_in2_rate * 0.01;

        // Motor and boat performance
        score += entry.motor_in2_rate * 0.02;
        score += entry.boat_in2_rate * 0.02;

        // Class bonus
        let class_bonus = match entry.racer_class.as_str() {
            "A1" => 2.0,
            "A2" => 1.5,
            "B1" => 1.0,
            "B2" => 0.5,
            _ => 1.0,
        };
        score += class_bonus;

        score.max(0.1)
    }

    /// Generate position probability distribution from win probability
    fn generate_position_dist(&self, win_prob: f64) -> [f64; 6] {
        let p1 = win_prob;
        let remaining = 1.0 - p1;

        // Decreasing probabilities for worse positions
        let ratios = [0.0, 0.35, 0.25, 0.20, 0.12, 0.08];

        [
            p1,
            remaining * ratios[1],
            remaining * ratios[2],
            remaining * ratios[3],
            remaining * ratios[4],
            remaining * ratios[5],
        ]
    }

    /// Normalize probabilities so each position sums to 1 across boats
    fn normalize_probs(&self, probs: &mut Vec<PositionProb>) {
        for pos in 0..6 {
            let sum: f64 = probs.iter().map(|p| p.probs[pos]).sum();
            if sum > 0.0 {
                for p in probs.iter_mut() {
                    p.probs[pos] /= sum;
                }
            }
        }
    }

    /// Calculate exacta (1st-2nd) probabilities
    pub fn calculate_exacta_probs(&self, position_probs: &[PositionProb]) -> Vec<ExactaPrediction> {
        let mut predictions = Vec::with_capacity(30);

        for first in position_probs {
            let p_first = first.probs[0]; // P(1st)

            for second in position_probs {
                if first.boat_no == second.boat_no {
                    continue;
                }

                let p_second = second.probs[1]; // P(2nd)

                // Conditional probability: P(B=2nd | A=1st)
                // ≈ P(B=2nd) / (1 - P(B=1st))
                let p_second_given_first = p_second / (1.0 - second.probs[0]).max(0.01);

                // Exacta probability
                let probability = p_first * p_second_given_first;

                predictions.push(ExactaPrediction {
                    first: first.boat_no,
                    second: second.boat_no,
                    probability,
                    odds: None,
                    expected_value: None,
                    is_value_bet: false,
                });
            }
        }

        // Sort by probability descending
        predictions.sort_by(|a, b| b.probability.partial_cmp(&a.probability).unwrap());

        predictions
    }
}

impl Default for Predictor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_entries() -> Vec<RacerEntry> {
        vec![
            RacerEntry {
                boat_no: 1, racer_id: 3527, racer_name: "中嶋誠一".to_string(),
                age: 53, weight: 51, racer_class: "A2".to_string(),
                national_win_rate: 5.10, national_in2_rate: 30.40,
                local_win_rate: 5.91, local_in2_rate: 43.18,
                motor_no: 55, motor_in2_rate: 15.87, boat_no_equip: 78, boat_in2_rate: 33.33,
            },
            RacerEntry {
                boat_no: 2, racer_id: 5036, racer_name: "福田翔吾".to_string(),
                age: 25, weight: 52, racer_class: "B1".to_string(),
                national_win_rate: 5.09, national_in2_rate: 32.63,
                local_win_rate: 4.33, local_in2_rate: 33.33,
                motor_no: 51, motor_in2_rate: 28.40, boat_no_equip: 85, boat_in2_rate: 24.72,
            },
            RacerEntry {
                boat_no: 3, racer_id: 5160, racer_name: "藤森陸斗".to_string(),
                age: 24, weight: 54, racer_class: "B1".to_string(),
                national_win_rate: 4.18, national_in2_rate: 20.48,
                local_win_rate: 2.64, local_in2_rate: 4.00,
                motor_no: 27, motor_in2_rate: 54.17, boat_no_equip: 84, boat_in2_rate: 37.22,
            },
            RacerEntry {
                boat_no: 4, racer_id: 4861, racer_name: "田中宏樹".to_string(),
                age: 35, weight: 53, racer_class: "B1".to_string(),
                national_win_rate: 4.92, national_in2_rate: 26.03,
                local_win_rate: 4.96, local_in2_rate: 37.04,
                motor_no: 54, motor_in2_rate: 24.66, boat_no_equip: 72, boat_in2_rate: 29.73,
            },
            RacerEntry {
                boat_no: 5, racer_id: 4876, racer_name: "梅木敬太".to_string(),
                age: 29, weight: 53, racer_class: "B1".to_string(),
                national_win_rate: 3.66, national_in2_rate: 16.42,
                local_win_rate: 4.84, local_in2_rate: 31.58,
                motor_no: 26, motor_in2_rate: 34.12, boat_no_equip: 31, boat_in2_rate: 38.78,
            },
            RacerEntry {
                boat_no: 6, racer_id: 4097, racer_name: "貫地谷直人".to_string(),
                age: 41, weight: 55, racer_class: "B1".to_string(),
                national_win_rate: 4.26, national_in2_rate: 20.78,
                local_win_rate: 4.69, local_in2_rate: 28.57,
                motor_no: 18, motor_in2_rate: 38.37, boat_no_equip: 71, boat_in2_rate: 35.71,
            },
        ]
    }

    #[test]
    fn test_predict_positions() {
        let predictor = Predictor::new();
        let entries = sample_entries();
        let probs = predictor.predict_positions(&entries);

        assert_eq!(probs.len(), 6);

        // Each boat should have 6 position probabilities
        for p in &probs {
            assert_eq!(p.probs.len(), 6);
            // Probabilities should be positive
            for prob in p.probs {
                assert!(prob >= 0.0);
            }
        }

        // Boat 1 should have highest win probability (course advantage)
        let boat1_win = probs.iter().find(|p| p.boat_no == 1).unwrap().probs[0];
        let boat6_win = probs.iter().find(|p| p.boat_no == 6).unwrap().probs[0];
        assert!(boat1_win > boat6_win);
    }

    #[test]
    fn test_calculate_exacta_probs() {
        let predictor = Predictor::new();
        let entries = sample_entries();
        let position_probs = predictor.predict_positions(&entries);
        let exacta_probs = predictor.calculate_exacta_probs(&position_probs);

        // Should have 30 combinations (6 * 5)
        assert_eq!(exacta_probs.len(), 30);

        // All probabilities should be positive
        for pred in &exacta_probs {
            assert!(pred.probability > 0.0);
            assert!(pred.first != pred.second);
        }

        // Should be sorted by probability descending
        for i in 1..exacta_probs.len() {
            assert!(exacta_probs[i-1].probability >= exacta_probs[i].probability);
        }
    }
}
