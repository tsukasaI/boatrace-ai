use crate::models::{ExactaPrediction, PositionProb, RacerEntry, TrifectaPrediction};
use ort::{
    session::{builder::GraphOptimizationLevel, Session},
    value::Tensor,
};
use std::path::Path;
use tracing::info;

/// Number of position models (one per finishing position 1-6)
const NUM_MODELS: usize = 6;
/// Number of features expected by the model (9 base + 9 historical + 5 relative + 3 exhibition)
const NUM_FEATURES: usize = 26;

/// ONNX-based predictor for boat race outcomes
pub struct Predictor {
    sessions: Vec<Session>,
}

impl Predictor {
    /// Create a new predictor by loading ONNX models from directory
    pub fn new<P: AsRef<Path>>(model_dir: P) -> Result<Self, Box<dyn std::error::Error>> {
        let model_dir = model_dir.as_ref();
        let mut sessions = Vec::with_capacity(NUM_MODELS);

        for i in 1..=NUM_MODELS {
            let model_path = model_dir.join(format!("position_{}.onnx", i));
            info!("Loading model: {:?}", model_path);

            let session = Session::builder()?
                .with_optimization_level(GraphOptimizationLevel::Level3)?
                .commit_from_file(&model_path)?;

            sessions.push(session);
        }

        info!("Loaded {} ONNX models", sessions.len());
        Ok(Self { sessions })
    }

    /// Predict position probabilities for each boat
    ///
    /// Returns 6 boats × 6 positions probability matrix
    /// Returns Err if ONNX inference fails
    pub fn predict_positions(
        &mut self,
        entries: &[RacerEntry],
    ) -> Result<Vec<PositionProb>, Box<dyn std::error::Error>> {
        self.predict_positions_with_history(entries, None)
    }

    /// Predict position probabilities with real historical features
    ///
    /// # Arguments
    /// * `entries` - 6 racer entries for the race
    /// * `historical` - Optional historical features for each racer (indexed by position)
    pub fn predict_positions_with_history(
        &mut self,
        entries: &[RacerEntry],
        historical: Option<&[crate::data::HistoricalFeatures]>,
    ) -> Result<Vec<PositionProb>, Box<dyn std::error::Error>> {
        if entries.len() != 6 {
            return Err("Exactly 6 entries required".into());
        }

        // Create feature matrix (6 boats × 23 features)
        let features = self.extract_features_with_history(entries, historical);

        // Run inference for each position model
        let mut position_probs = vec![[0.0f64; 6]; 6]; // boats × positions

        for (pos_idx, session) in self.sessions.iter_mut().enumerate() {
            let input_vec: Vec<f32> = features.iter().map(|&x| x as f32).collect();
            let input_tensor = Tensor::from_array(([6usize, NUM_FEATURES], input_vec))?;

            let outputs = session.run(ort::inputs!["input" => input_tensor])?;

            // Extract predictions for each boat
            let (_, output_data) = outputs[0].try_extract_tensor::<f32>()?;

            for (boat_idx, value) in output_data.iter().enumerate() {
                if boat_idx < 6 {
                    position_probs[boat_idx][pos_idx] = *value as f64;
                }
            }
        }

        // Convert raw scores to probabilities using softmax
        let probs = self.softmax_normalize(position_probs);

        // Build result
        Ok(entries
            .iter()
            .enumerate()
            .map(|(i, e)| PositionProb {
                boat_no: e.boat_no,
                probs: probs[i],
            })
            .collect())
    }

    /// Extract features from race entries with optional real historical features
    ///
    /// Order: Base (9) + Historical (9) + Relative (5) = 23 features per boat
    fn extract_features_with_history(
        &self,
        entries: &[RacerEntry],
        historical: Option<&[crate::data::HistoricalFeatures]>,
    ) -> Vec<f64> {
        let mut features = Vec::with_capacity(6 * NUM_FEATURES);

        // Calculate averages for relative features
        let avg_win_rate: f64 =
            entries.iter().map(|e| e.national_win_rate).sum::<f64>() / entries.len() as f64;

        for (i, entry) in entries.iter().enumerate() {
            // 1. Base features (9)
            features.push(entry.national_win_rate);
            features.push(entry.national_in2_rate);
            features.push(entry.local_win_rate);
            features.push(entry.local_in2_rate);
            features.push(entry.age as f64);
            features.push(entry.weight as f64);
            features.push(Self::encode_class(&entry.racer_class));
            features.push(entry.motor_in2_rate);
            features.push(entry.boat_in2_rate);

            // 2. Historical features (9) - use real if available, otherwise proxy
            if let Some(hist_list) = historical {
                if let Some(hist) = hist_list.get(i) {
                    features.push(hist.recent_win_rate);
                    features.push(hist.recent_in2_rate);
                    features.push(hist.recent_in3_rate);
                    features.push(hist.recent_avg_rank);
                    features.push(hist.recent_avg_st);
                    features.push(hist.recent_race_count);
                    features.push(hist.local_recent_win_rate);
                    features.push(hist.local_race_count);
                    features.push(hist.course_win_rate);
                } else {
                    self.push_proxy_historical_features(entry, &mut features);
                }
            } else {
                self.push_proxy_historical_features(entry, &mut features);
            }

            // 3. Relative features (5)
            let win_rate_rank = Self::calculate_rank(entries, |e| e.national_win_rate, entry);
            let win_rate_diff = entry.national_win_rate - avg_win_rate;
            let motor_rate_rank = Self::calculate_rank(entries, |e| e.motor_in2_rate, entry);
            let boat_rate_rank = Self::calculate_rank(entries, |e| e.boat_in2_rate, entry);
            let course_advantage = Self::get_course_advantage(entry.boat_no);

            features.push(win_rate_rank);
            features.push(win_rate_diff);
            features.push(motor_rate_rank);
            features.push(boat_rate_rank);
            features.push(course_advantage);
        }

        features
    }

    /// Push proxy historical features derived from base features
    fn push_proxy_historical_features(&self, entry: &RacerEntry, features: &mut Vec<f64>) {
        let recent_win_rate = entry.national_win_rate / 100.0;
        let recent_in2_rate = entry.national_in2_rate / 100.0;
        let recent_in3_rate = (entry.national_in2_rate + 15.0).min(100.0) / 100.0;
        let recent_avg_rank = 7.0 - entry.national_win_rate / 2.0;
        let recent_avg_st = 0.15;
        let recent_race_count = 30.0;
        let local_recent_win_rate = entry.local_win_rate / 100.0;
        let local_race_count = 10.0;
        let course_win_rate = Self::get_course_advantage(entry.boat_no);

        features.push(recent_win_rate);
        features.push(recent_in2_rate);
        features.push(recent_in3_rate);
        features.push(recent_avg_rank);
        features.push(recent_avg_st);
        features.push(recent_race_count);
        features.push(local_recent_win_rate);
        features.push(local_race_count);
        features.push(course_win_rate);
    }

    /// Extract features from race entries (proxy historical features)
    #[allow(dead_code)]
    fn extract_features(&self, entries: &[RacerEntry]) -> Vec<f64> {
        self.extract_features_with_history(entries, None)
    }

    /// Encode racer class to numeric value
    fn encode_class(class: &str) -> f64 {
        match class {
            "A1" => 4.0,
            "A2" => 3.0,
            "B1" => 2.0,
            "B2" => 1.0,
            _ => 2.0,
        }
    }

    /// Calculate rank (1-6) for a given metric
    fn calculate_rank<F>(entries: &[RacerEntry], metric: F, target: &RacerEntry) -> f64
    where
        F: Fn(&RacerEntry) -> f64,
    {
        let target_value = metric(target);
        let rank = entries.iter().filter(|e| metric(e) > target_value).count() + 1;
        rank as f64
    }

    /// Get historical course advantage value
    fn get_course_advantage(boat_no: u8) -> f64 {
        match boat_no {
            1 => 0.55,
            2 => 0.14,
            3 => 0.12,
            4 => 0.10,
            5 => 0.06,
            6 => 0.03,
            _ => 0.10,
        }
    }

    /// Apply softmax normalization across positions for each boat,
    /// then normalize across boats for each position
    fn softmax_normalize(&self, scores: Vec<[f64; 6]>) -> Vec<[f64; 6]> {
        let mut probs = vec![[0.0f64; 6]; 6];

        // First, apply softmax for each boat across positions
        for (i, boat_scores) in scores.iter().enumerate() {
            let max_score = boat_scores.iter().cloned().fold(f64::MIN, f64::max);
            let exp_sum: f64 = boat_scores.iter().map(|&s| (s - max_score).exp()).sum();

            for (j, &score) in boat_scores.iter().enumerate() {
                probs[i][j] = (score - max_score).exp() / exp_sum;
            }
        }

        // Then normalize so each position sums to 1 across boats
        for pos in 0..6 {
            let sum: f64 = probs.iter().map(|p| p[pos]).sum();
            if sum > 0.0 {
                for prob in &mut probs {
                    prob[pos] /= sum;
                }
            }
        }

        probs
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

    /// Calculate trifecta (3連単) probabilities - 120 combinations
    ///
    /// Formula: P(1st) × P(2nd|1st) × P(3rd|1st,2nd)
    pub fn calculate_trifecta_probs(
        &self,
        position_probs: &[PositionProb],
    ) -> Vec<TrifectaPrediction> {
        let mut predictions = Vec::with_capacity(120);

        for first in position_probs {
            let p_first = first.probs[0]; // P(1st)

            for second in position_probs {
                if first.boat_no == second.boat_no {
                    continue;
                }

                // Conditional probability: P(B=2nd | A=1st)
                let p_second_given_first = second.probs[1] / (1.0 - second.probs[0]).max(0.01);

                for third in position_probs {
                    if third.boat_no == first.boat_no || third.boat_no == second.boat_no {
                        continue;
                    }

                    // Conditional probability: P(C=3rd | A=1st, B=2nd)
                    // Approximate: P(C=3rd) / P(C not in top 2)
                    let p_third = third.probs[2];
                    let p_third_not_top2 = (1.0 - third.probs[0] - third.probs[1]).max(0.01);
                    let p_third_given = p_third / p_third_not_top2;

                    // Trifecta probability
                    let probability = p_first * p_second_given_first * p_third_given;

                    predictions.push(TrifectaPrediction {
                        first: first.boat_no,
                        second: second.boat_no,
                        third: third.boat_no,
                        probability,
                        odds: None,
                        expected_value: None,
                        is_value_bet: false,
                    });
                }
            }
        }

        // Sort by probability descending
        predictions.sort_by(|a, b| b.probability.partial_cmp(&a.probability).unwrap());

        predictions
    }
}

/// Fallback predictor using heuristics (when ONNX models not available)
pub struct FallbackPredictor {
    course_advantage: [f64; 6],
}

impl FallbackPredictor {
    pub fn new() -> Self {
        Self {
            course_advantage: [0.55, 0.14, 0.12, 0.10, 0.06, 0.03],
        }
    }

    pub fn predict_positions(&self, entries: &[RacerEntry]) -> Vec<PositionProb> {
        let mut probs = Vec::with_capacity(6);

        let scores: Vec<f64> = entries.iter().map(|e| self.calculate_score(e)).collect();
        let total_score: f64 = scores.iter().sum();

        for (i, entry) in entries.iter().enumerate() {
            let base_prob = if total_score > 0.0 {
                scores[i] / total_score
            } else {
                1.0 / 6.0
            };

            let course_idx = (entry.boat_no - 1) as usize;
            let win_prob = 0.6 * base_prob + 0.4 * self.course_advantage[course_idx];
            let position_probs = self.generate_position_dist(win_prob);

            probs.push(PositionProb {
                boat_no: entry.boat_no,
                probs: position_probs,
            });
        }

        self.normalize_probs(&mut probs);
        probs
    }

    fn calculate_score(&self, entry: &RacerEntry) -> f64 {
        let mut score = 0.0;
        score += entry.national_win_rate * 0.3;
        score += entry.local_win_rate * 0.2;
        score += entry.national_in2_rate * 0.01;
        score += entry.local_in2_rate * 0.01;
        score += entry.motor_in2_rate * 0.02;
        score += entry.boat_in2_rate * 0.02;

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

    fn generate_position_dist(&self, win_prob: f64) -> [f64; 6] {
        let p1 = win_prob;
        let remaining = 1.0 - p1;
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

    fn normalize_probs(&self, probs: &mut [PositionProb]) {
        for pos in 0..6 {
            let sum: f64 = probs.iter().map(|p| p.probs[pos]).sum();
            if sum > 0.0 {
                for p in probs.iter_mut() {
                    p.probs[pos] /= sum;
                }
            }
        }
    }

    pub fn calculate_exacta_probs(&self, position_probs: &[PositionProb]) -> Vec<ExactaPrediction> {
        let mut predictions = Vec::with_capacity(30);

        for first in position_probs {
            let p_first = first.probs[0];
            for second in position_probs {
                if first.boat_no == second.boat_no {
                    continue;
                }
                let p_second = second.probs[1];
                let p_second_given_first = p_second / (1.0 - second.probs[0]).max(0.01);
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

        predictions.sort_by(|a, b| b.probability.partial_cmp(&a.probability).unwrap());
        predictions
    }

    /// Calculate trifecta (3連単) probabilities - 120 combinations
    pub fn calculate_trifecta_probs(
        &self,
        position_probs: &[PositionProb],
    ) -> Vec<TrifectaPrediction> {
        let mut predictions = Vec::with_capacity(120);

        for first in position_probs {
            let p_first = first.probs[0];

            for second in position_probs {
                if first.boat_no == second.boat_no {
                    continue;
                }

                let p_second_given_first = second.probs[1] / (1.0 - second.probs[0]).max(0.01);

                for third in position_probs {
                    if third.boat_no == first.boat_no || third.boat_no == second.boat_no {
                        continue;
                    }

                    let p_third = third.probs[2];
                    let p_third_not_top2 = (1.0 - third.probs[0] - third.probs[1]).max(0.01);
                    let p_third_given = p_third / p_third_not_top2;

                    let probability = p_first * p_second_given_first * p_third_given;

                    predictions.push(TrifectaPrediction {
                        first: first.boat_no,
                        second: second.boat_no,
                        third: third.boat_no,
                        probability,
                        odds: None,
                        expected_value: None,
                        is_value_bet: false,
                    });
                }
            }
        }

        predictions.sort_by(|a, b| b.probability.partial_cmp(&a.probability).unwrap());
        predictions
    }
}

impl Default for FallbackPredictor {
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
                boat_no: 1,
                racer_id: 3527,
                racer_name: "中嶋誠一".to_string(),
                age: 53,
                weight: 51,
                racer_class: "A2".to_string(),
                national_win_rate: 5.10,
                national_in2_rate: 30.40,
                local_win_rate: 5.91,
                local_in2_rate: 43.18,
                motor_no: 55,
                motor_in2_rate: 15.87,
                boat_no_equip: 78,
                boat_in2_rate: 33.33,
            },
            RacerEntry {
                boat_no: 2,
                racer_id: 5036,
                racer_name: "福田翔吾".to_string(),
                age: 25,
                weight: 52,
                racer_class: "B1".to_string(),
                national_win_rate: 5.09,
                national_in2_rate: 32.63,
                local_win_rate: 4.33,
                local_in2_rate: 33.33,
                motor_no: 51,
                motor_in2_rate: 28.40,
                boat_no_equip: 85,
                boat_in2_rate: 24.72,
            },
            RacerEntry {
                boat_no: 3,
                racer_id: 5160,
                racer_name: "藤森陸斗".to_string(),
                age: 24,
                weight: 54,
                racer_class: "B1".to_string(),
                national_win_rate: 4.18,
                national_in2_rate: 20.48,
                local_win_rate: 2.64,
                local_in2_rate: 4.00,
                motor_no: 27,
                motor_in2_rate: 54.17,
                boat_no_equip: 84,
                boat_in2_rate: 37.22,
            },
            RacerEntry {
                boat_no: 4,
                racer_id: 4861,
                racer_name: "田中宏樹".to_string(),
                age: 35,
                weight: 53,
                racer_class: "B1".to_string(),
                national_win_rate: 4.92,
                national_in2_rate: 26.03,
                local_win_rate: 4.96,
                local_in2_rate: 37.04,
                motor_no: 54,
                motor_in2_rate: 24.66,
                boat_no_equip: 72,
                boat_in2_rate: 29.73,
            },
            RacerEntry {
                boat_no: 5,
                racer_id: 4876,
                racer_name: "梅木敬太".to_string(),
                age: 29,
                weight: 53,
                racer_class: "B1".to_string(),
                national_win_rate: 3.66,
                national_in2_rate: 16.42,
                local_win_rate: 4.84,
                local_in2_rate: 31.58,
                motor_no: 26,
                motor_in2_rate: 34.12,
                boat_no_equip: 31,
                boat_in2_rate: 38.78,
            },
            RacerEntry {
                boat_no: 6,
                racer_id: 4097,
                racer_name: "貫地谷直人".to_string(),
                age: 41,
                weight: 55,
                racer_class: "B1".to_string(),
                national_win_rate: 4.26,
                national_in2_rate: 20.78,
                local_win_rate: 4.69,
                local_in2_rate: 28.57,
                motor_no: 18,
                motor_in2_rate: 38.37,
                boat_no_equip: 71,
                boat_in2_rate: 35.71,
            },
        ]
    }

    #[test]
    fn test_fallback_predict_positions() {
        let predictor = FallbackPredictor::new();
        let entries = sample_entries();
        let probs = predictor.predict_positions(&entries);

        assert_eq!(probs.len(), 6);

        for p in &probs {
            assert_eq!(p.probs.len(), 6);
            for prob in p.probs {
                assert!(prob >= 0.0);
            }
        }

        let boat1_win = probs.iter().find(|p| p.boat_no == 1).unwrap().probs[0];
        let boat6_win = probs.iter().find(|p| p.boat_no == 6).unwrap().probs[0];
        assert!(boat1_win > boat6_win);
    }

    #[test]
    fn test_fallback_calculate_exacta_probs() {
        let predictor = FallbackPredictor::new();
        let entries = sample_entries();
        let position_probs = predictor.predict_positions(&entries);
        let exacta_probs = predictor.calculate_exacta_probs(&position_probs);

        assert_eq!(exacta_probs.len(), 30);

        for pred in &exacta_probs {
            assert!(pred.probability > 0.0);
            assert!(pred.first != pred.second);
        }

        for i in 1..exacta_probs.len() {
            assert!(exacta_probs[i - 1].probability >= exacta_probs[i].probability);
        }
    }

    #[test]
    fn test_fallback_calculate_trifecta_probs() {
        let predictor = FallbackPredictor::new();
        let entries = sample_entries();
        let position_probs = predictor.predict_positions(&entries);
        let trifecta_probs = predictor.calculate_trifecta_probs(&position_probs);

        // 6 × 5 × 4 = 120 combinations
        assert_eq!(trifecta_probs.len(), 120);

        for pred in &trifecta_probs {
            assert!(pred.probability > 0.0);
            assert!(pred.first != pred.second);
            assert!(pred.first != pred.third);
            assert!(pred.second != pred.third);
        }

        // Check sorted by probability descending
        for i in 1..trifecta_probs.len() {
            assert!(trifecta_probs[i - 1].probability >= trifecta_probs[i].probability);
        }
    }

    #[test]
    fn test_trifecta_probabilities_sum() {
        let predictor = FallbackPredictor::new();
        let entries = sample_entries();
        let position_probs = predictor.predict_positions(&entries);
        let trifecta_probs = predictor.calculate_trifecta_probs(&position_probs);

        // Sum of all trifecta probabilities should be close to 1
        // (allowing for approximation errors in conditional probability)
        let total_prob: f64 = trifecta_probs.iter().map(|p| p.probability).sum();
        assert!(total_prob > 0.5, "Total probability {} too low", total_prob);
        assert!(
            total_prob < 2.0,
            "Total probability {} too high",
            total_prob
        );
    }
}
