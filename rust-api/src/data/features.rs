//! Feature Engineering
//!
//! Generate features from racer, motor, boat statistics and past performance

use crate::data::ProgramEntry;
use serde::{Deserialize, Serialize};

/// Class rank encoding
fn encode_class(class: &str) -> f64 {
    match class {
        "A1" => 4.0,
        "A2" => 3.0,
        "B1" => 2.0,
        "B2" => 1.0,
        _ => 0.0,
    }
}

/// Branch/Region encoding (grouped by geography)
fn encode_branch(branch: &str) -> f64 {
    match branch {
        // Kanto
        "群馬" | "埼玉" | "東京" => 1.0,
        // Tokai
        "静岡" | "愛知" | "三重" => 2.0,
        // Kinki
        "滋賀" | "大阪" | "兵庫" => 3.0,
        // Chugoku/Shikoku
        "岡山" | "広島" | "山口" | "徳島" | "香川" => 4.0,
        // Kyushu
        "福岡" | "佐賀" | "長崎" | "大分" => 5.0,
        _ => 0.0,
    }
}

/// Course advantage by boat number (lane 1 is most advantageous)
fn course_advantage(boat_no: u8) -> f64 {
    match boat_no {
        1 => 0.55,
        2 => 0.14,
        3 => 0.12,
        4 => 0.10,
        5 => 0.06,
        6 => 0.03,
        _ => 0.0,
    }
}

/// Base features extracted from program entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaseFeatures {
    pub boat_no: u8,
    pub racer_id: u32,
    // Racer features
    pub national_win_rate: f64,
    pub national_in2_rate: f64,
    pub local_win_rate: f64,
    pub local_in2_rate: f64,
    pub age: f64,
    pub weight: f64,
    pub class_encoded: f64,
    pub branch_encoded: f64,
    // Equipment features
    pub motor_in2_rate: f64,
    pub boat_in2_rate: f64,
}

/// Relative features within a race
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelativeFeatures {
    pub win_rate_rank: f64,
    pub win_rate_diff_from_avg: f64,
    pub motor_rate_rank: f64,
    pub boat_rate_rank: f64,
    pub course_advantage: f64,
}

/// Exhibition time features (from pre-race exhibition)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExhibitionFeatures {
    pub exhibition_time: f64,      // Exhibition time in seconds (e.g., 6.80)
    pub exhibition_time_rank: f64, // Rank within race (1=fastest, 6=slowest)
    pub exhibition_time_diff: f64, // Difference from race average
}

impl Default for ExhibitionFeatures {
    /// Default values when exhibition time is not available
    fn default() -> Self {
        Self {
            exhibition_time: 6.80,      // Average exhibition time
            exhibition_time_rank: 3.5,  // Middle rank
            exhibition_time_diff: 0.0,  // No difference from average
        }
    }
}

/// Historical features from past performance (optional)
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct HistoricalFeatures {
    pub recent_win_rate: f64,
    pub recent_in2_rate: f64,
    pub recent_in3_rate: f64,
    pub recent_avg_rank: f64,
    pub recent_avg_st: f64,
    pub recent_race_count: f64,
    pub local_recent_win_rate: f64,
    pub local_race_count: f64,
    pub course_win_rate: f64,
    // New historical features
    pub course_in2_rate: f64,
    pub st_consistency: f64,
    pub flying_start_rate: f64,
    pub late_start_rate: f64,
    pub avg_course_diff: f64,
    pub inside_take_rate: f64,
    pub weighted_recent_win: f64,
}

/// Race context features
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RaceContextFeatures {
    pub race_grade: f64,  // 1=qualifying, 2=semi, 3=semi-final, 4=final
    pub is_final: f64,    // 1 if race_grade >= 3, else 0
}

/// Interaction features (combinations of base features)
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct InteractionFeatures {
    pub class_x_course: f64,      // class_encoded * course_advantage
    pub motor_x_exhibition: f64,  // motor_in2_rate * exhibition_score / 100
    pub equipment_score: f64,     // (motor_in2_rate + boat_in2_rate) / 2
    pub equipment_rank: f64,      // rank within race (1=best)
    pub favorite_score: f64,      // combined strength indicator
    pub upset_potential: f64,     // high class but outside course
}

/// Complete feature set for a single entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RacerFeatures {
    pub boat_no: u8,
    pub racer_id: u32,
    pub base: BaseFeatures,
    pub relative: RelativeFeatures,
    pub historical: Option<HistoricalFeatures>,
    pub exhibition: ExhibitionFeatures,
    pub context: RaceContextFeatures,
    pub interaction: InteractionFeatures,
}

impl RacerFeatures {
    /// Convert features to a flat vector for model input
    ///
    /// Order matches ONNX model (42 features):
    /// Base (10) + Historical (16) + Relative (5) + Exhibition (3) + Context (2) + Interaction (6)
    pub fn to_vec(&self) -> Vec<f64> {
        let mut features = vec![
            // Base features (10)
            self.base.national_win_rate,
            self.base.national_in2_rate,
            self.base.local_win_rate,
            self.base.local_in2_rate,
            self.base.age,
            self.base.weight,
            self.base.class_encoded,
            self.base.branch_encoded,
            self.base.motor_in2_rate,
            self.base.boat_in2_rate,
        ];

        // Historical features (16) - must come before relative to match ONNX model order
        if let Some(ref hist) = self.historical {
            features.extend([
                hist.recent_win_rate,
                hist.recent_in2_rate,
                hist.recent_in3_rate,
                hist.recent_avg_rank,
                hist.recent_avg_st,
                hist.recent_race_count,
                hist.local_recent_win_rate,
                hist.local_race_count,
                hist.course_win_rate,
                hist.course_in2_rate,
                hist.st_consistency,
                hist.flying_start_rate,
                hist.late_start_rate,
                hist.avg_course_diff,
                hist.inside_take_rate,
                hist.weighted_recent_win,
            ]);
        }

        // Relative features (5)
        features.extend([
            self.relative.win_rate_rank,
            self.relative.win_rate_diff_from_avg,
            self.relative.motor_rate_rank,
            self.relative.boat_rate_rank,
            self.relative.course_advantage,
        ]);

        // Exhibition time features (3)
        features.extend([
            self.exhibition.exhibition_time,
            self.exhibition.exhibition_time_rank,
            self.exhibition.exhibition_time_diff,
        ]);

        // Race context features (2)
        features.extend([self.context.race_grade, self.context.is_final]);

        // Interaction features (6)
        features.extend([
            self.interaction.class_x_course,
            self.interaction.motor_x_exhibition,
            self.interaction.equipment_score,
            self.interaction.equipment_rank,
            self.interaction.favorite_score,
            self.interaction.upset_potential,
        ]);

        features
    }
}

/// Feature engineering for race entries
pub struct FeatureEngineering;

impl FeatureEngineering {
    /// Create base features from a program entry
    pub fn create_base_features(entry: &ProgramEntry) -> BaseFeatures {
        BaseFeatures {
            boat_no: entry.boat_no,
            racer_id: entry.racer_id,
            national_win_rate: entry.national_win_rate,
            national_in2_rate: entry.national_in2_rate,
            local_win_rate: entry.local_win_rate,
            local_in2_rate: entry.local_in2_rate,
            age: entry.age as f64,
            weight: entry.weight as f64,
            class_encoded: encode_class(&entry.racer_class),
            branch_encoded: encode_branch(&entry.branch),
            motor_in2_rate: entry.motor_in2_rate,
            boat_in2_rate: entry.boat_in2_rate,
        }
    }

    /// Create relative features for all entries in a race
    pub fn create_relative_features(entries: &[ProgramEntry]) -> Vec<RelativeFeatures> {
        if entries.is_empty() {
            return Vec::new();
        }

        // Calculate race averages
        let avg_win_rate: f64 =
            entries.iter().map(|e| e.national_win_rate).sum::<f64>() / entries.len() as f64;

        // Sort indices by various rates for ranking
        let mut win_rate_order: Vec<(usize, f64)> = entries
            .iter()
            .enumerate()
            .map(|(i, e)| (i, e.national_win_rate))
            .collect();
        win_rate_order.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut motor_rate_order: Vec<(usize, f64)> = entries
            .iter()
            .enumerate()
            .map(|(i, e)| (i, e.motor_in2_rate))
            .collect();
        motor_rate_order.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut boat_rate_order: Vec<(usize, f64)> = entries
            .iter()
            .enumerate()
            .map(|(i, e)| (i, e.boat_in2_rate))
            .collect();
        boat_rate_order.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Create rank maps (1-based, lower is better)
        let mut win_rate_ranks = vec![0.0; entries.len()];
        let mut motor_rate_ranks = vec![0.0; entries.len()];
        let mut boat_rate_ranks = vec![0.0; entries.len()];

        for (rank, (idx, _)) in win_rate_order.iter().enumerate() {
            win_rate_ranks[*idx] = (rank + 1) as f64;
        }
        for (rank, (idx, _)) in motor_rate_order.iter().enumerate() {
            motor_rate_ranks[*idx] = (rank + 1) as f64;
        }
        for (rank, (idx, _)) in boat_rate_order.iter().enumerate() {
            boat_rate_ranks[*idx] = (rank + 1) as f64;
        }

        // Build relative features for each entry
        entries
            .iter()
            .enumerate()
            .map(|(i, entry)| RelativeFeatures {
                win_rate_rank: win_rate_ranks[i],
                win_rate_diff_from_avg: entry.national_win_rate - avg_win_rate,
                motor_rate_rank: motor_rate_ranks[i],
                boat_rate_rank: boat_rate_ranks[i],
                course_advantage: course_advantage(entry.boat_no),
            })
            .collect()
    }

    /// Create all features for a race (base + relative + default exhibition)
    pub fn create_race_features(entries: &[ProgramEntry]) -> Vec<RacerFeatures> {
        Self::create_race_features_with_exhibition(entries, None)
    }

    /// Create all features for a race with exhibition times
    ///
    /// # Arguments
    /// * `entries` - Program entries for the race
    /// * `exhibition_times` - Optional exhibition times for each boat (indexed by boat_no - 1)
    pub fn create_race_features_with_exhibition(
        entries: &[ProgramEntry],
        exhibition_times: Option<&[f64; 6]>,
    ) -> Vec<RacerFeatures> {
        let base_features: Vec<BaseFeatures> =
            entries.iter().map(Self::create_base_features).collect();

        let relative_features = Self::create_relative_features(entries);

        // Calculate exhibition features if times are provided
        let exhibition_features: Vec<ExhibitionFeatures> = if let Some(times) = exhibition_times {
            Self::create_exhibition_features(entries, times)
        } else {
            // Use default values for all entries
            entries
                .iter()
                .map(|_| ExhibitionFeatures::default())
                .collect()
        };

        // Calculate interaction features
        let interaction_features =
            Self::create_interaction_features(&base_features, &relative_features, &exhibition_features);

        base_features
            .into_iter()
            .zip(relative_features)
            .zip(exhibition_features)
            .zip(interaction_features)
            .map(|(((base, relative), exhibition), interaction)| RacerFeatures {
                boat_no: base.boat_no,
                racer_id: base.racer_id,
                base,
                relative,
                historical: None,
                exhibition,
                context: RaceContextFeatures::default(), // Set separately if race_type available
                interaction,
            })
            .collect()
    }

    /// Create interaction features
    fn create_interaction_features(
        base_features: &[BaseFeatures],
        relative_features: &[RelativeFeatures],
        exhibition_features: &[ExhibitionFeatures],
    ) -> Vec<InteractionFeatures> {
        if base_features.is_empty() {
            return Vec::new();
        }

        // Calculate equipment scores and ranks
        let equipment_scores: Vec<f64> = base_features
            .iter()
            .map(|b| (b.motor_in2_rate + b.boat_in2_rate) / 2.0)
            .collect();

        // Sort for ranking (higher is better)
        let mut equip_order: Vec<(usize, f64)> =
            equipment_scores.iter().enumerate().map(|(i, &s)| (i, s)).collect();
        equip_order.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut equipment_ranks = vec![0.0; base_features.len()];
        for (rank, (idx, _)) in equip_order.iter().enumerate() {
            equipment_ranks[*idx] = (rank + 1) as f64;
        }

        base_features
            .iter()
            .zip(relative_features.iter())
            .zip(exhibition_features.iter())
            .enumerate()
            .map(|(i, ((base, relative), exhibition))| {
                // class × course advantage
                let class_x_course = base.class_encoded * relative.course_advantage;

                // motor × exhibition score (normalized)
                let exhibition_score = 7.0 - exhibition.exhibition_time.clamp(6.5, 7.5);
                let motor_x_exhibition = base.motor_in2_rate * exhibition_score / 100.0;

                // Favorite score (combined strength indicator)
                let favorite_score = (base.class_encoded / 4.0
                    + (7.0 - relative.win_rate_rank) / 6.0
                    + (7.0 - equipment_ranks[i]) / 6.0
                    + relative.course_advantage)
                    / 4.0;

                // Upset potential (high class but outside course)
                let upset_potential = base.class_encoded * (1.0 - relative.course_advantage);

                InteractionFeatures {
                    class_x_course,
                    motor_x_exhibition,
                    equipment_score: equipment_scores[i],
                    equipment_rank: equipment_ranks[i],
                    favorite_score,
                    upset_potential,
                }
            })
            .collect()
    }

    /// Create exhibition features from exhibition times
    fn create_exhibition_features(
        entries: &[ProgramEntry],
        times: &[f64; 6],
    ) -> Vec<ExhibitionFeatures> {
        // Calculate average exhibition time
        let valid_times: Vec<f64> = entries
            .iter()
            .map(|e| times[(e.boat_no - 1) as usize])
            .filter(|&t| t > 0.0)
            .collect();

        let avg_time = if valid_times.is_empty() {
            6.80 // Default average
        } else {
            valid_times.iter().sum::<f64>() / valid_times.len() as f64
        };

        // Sort entries by exhibition time for ranking (lower is better)
        let mut time_order: Vec<(usize, f64)> = entries
            .iter()
            .enumerate()
            .map(|(i, e)| (i, times[(e.boat_no - 1) as usize]))
            .collect();
        time_order.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Create rank map (1-based, lower time = better rank)
        let mut ranks = vec![0.0; entries.len()];
        for (rank, (idx, _)) in time_order.iter().enumerate() {
            ranks[*idx] = (rank + 1) as f64;
        }

        // Build exhibition features
        entries
            .iter()
            .enumerate()
            .map(|(i, e)| {
                let time = times[(e.boat_no - 1) as usize];
                ExhibitionFeatures {
                    exhibition_time: if time > 0.0 { time } else { 6.80 },
                    exhibition_time_rank: ranks[i],
                    exhibition_time_diff: if time > 0.0 { time - avg_time } else { 0.0 },
                }
            })
            .collect()
    }
}

/// Get feature column names for base + relative + exhibition + context + interaction features (26 features without historical)
pub fn get_base_feature_names() -> Vec<&'static str> {
    vec![
        // Base features (10)
        "national_win_rate",
        "national_in2_rate",
        "local_win_rate",
        "local_in2_rate",
        "age",
        "weight",
        "class_encoded",
        "branch_encoded",
        "motor_in2_rate",
        "boat_in2_rate",
        // Relative features (5)
        "win_rate_rank",
        "win_rate_diff_from_avg",
        "motor_rate_rank",
        "boat_rate_rank",
        "course_advantage",
        // Exhibition features (3)
        "exhibition_time",
        "exhibition_time_rank",
        "exhibition_time_diff",
        // Race context features (2)
        "race_grade",
        "is_final",
        // Interaction features (6)
        "class_x_course",
        "motor_x_exhibition",
        "equipment_score",
        "equipment_rank",
        "favorite_score",
        "upset_potential",
    ]
}

/// Get all feature column names including historical (42 features)
/// Order: Base (10) + Historical (16) + Relative (5) + Exhibition (3) + Context (2) + Interaction (6)
pub fn get_all_feature_names() -> Vec<&'static str> {
    vec![
        // Base features (10)
        "national_win_rate",
        "national_in2_rate",
        "local_win_rate",
        "local_in2_rate",
        "age",
        "weight",
        "class_encoded",
        "branch_encoded",
        "motor_in2_rate",
        "boat_in2_rate",
        // Historical features (16)
        "recent_win_rate",
        "recent_in2_rate",
        "recent_in3_rate",
        "recent_avg_rank",
        "recent_avg_st",
        "recent_race_count",
        "local_recent_win_rate",
        "local_race_count",
        "course_win_rate",
        "course_in2_rate",
        "st_consistency",
        "flying_start_rate",
        "late_start_rate",
        "avg_course_diff",
        "inside_take_rate",
        "weighted_recent_win",
        // Relative features (5)
        "win_rate_rank",
        "win_rate_diff_from_avg",
        "motor_rate_rank",
        "boat_rate_rank",
        "course_advantage",
        // Exhibition features (3)
        "exhibition_time",
        "exhibition_time_rank",
        "exhibition_time_diff",
        // Race context features (2)
        "race_grade",
        "is_final",
        // Interaction features (6)
        "class_x_course",
        "motor_x_exhibition",
        "equipment_score",
        "equipment_rank",
        "favorite_score",
        "upset_potential",
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_entries() -> Vec<ProgramEntry> {
        vec![
            ProgramEntry {
                date: 20240115,
                stadium_code: 23,
                race_no: 1,
                boat_no: 1,
                racer_id: 1001,
                racer_name: "Test A".to_string(),
                age: 30,
                branch: "Tokyo".to_string(),
                weight: 52,
                racer_class: "A1".to_string(),
                national_win_rate: 7.0,
                national_in2_rate: 50.0,
                local_win_rate: 7.5,
                local_in2_rate: 55.0,
                motor_no: 10,
                motor_in2_rate: 40.0,
                boat_no_equip: 20,
                boat_in2_rate: 35.0,
            },
            ProgramEntry {
                date: 20240115,
                stadium_code: 23,
                race_no: 1,
                boat_no: 2,
                racer_id: 1002,
                racer_name: "Test B".to_string(),
                age: 25,
                branch: "Osaka".to_string(),
                weight: 50,
                racer_class: "B1".to_string(),
                national_win_rate: 5.0,
                national_in2_rate: 35.0,
                local_win_rate: 5.5,
                local_in2_rate: 40.0,
                motor_no: 11,
                motor_in2_rate: 45.0,
                boat_no_equip: 21,
                boat_in2_rate: 38.0,
            },
            ProgramEntry {
                date: 20240115,
                stadium_code: 23,
                race_no: 1,
                boat_no: 3,
                racer_id: 1003,
                racer_name: "Test C".to_string(),
                age: 35,
                branch: "Nagoya".to_string(),
                weight: 54,
                racer_class: "A2".to_string(),
                national_win_rate: 6.0,
                national_in2_rate: 42.0,
                local_win_rate: 6.2,
                local_in2_rate: 45.0,
                motor_no: 12,
                motor_in2_rate: 38.0,
                boat_no_equip: 22,
                boat_in2_rate: 42.0,
            },
        ]
    }

    #[test]
    fn test_encode_class() {
        assert_eq!(encode_class("A1"), 4.0);
        assert_eq!(encode_class("A2"), 3.0);
        assert_eq!(encode_class("B1"), 2.0);
        assert_eq!(encode_class("B2"), 1.0);
        assert_eq!(encode_class("unknown"), 0.0);
    }

    #[test]
    fn test_course_advantage() {
        assert!((course_advantage(1) - 0.55).abs() < 0.01);
        assert!((course_advantage(6) - 0.03).abs() < 0.01);
        assert_eq!(course_advantage(7), 0.0);
    }

    #[test]
    fn test_create_base_features() {
        let entries = create_test_entries();
        let base = FeatureEngineering::create_base_features(&entries[0]);

        assert_eq!(base.boat_no, 1);
        assert_eq!(base.racer_id, 1001);
        assert!((base.national_win_rate - 7.0).abs() < 0.01);
        assert!((base.class_encoded - 4.0).abs() < 0.01); // A1
        assert!((base.age - 30.0).abs() < 0.01);
    }

    #[test]
    fn test_create_relative_features() {
        let entries = create_test_entries();
        let relative = FeatureEngineering::create_relative_features(&entries);

        assert_eq!(relative.len(), 3);

        // Entry 0 has highest win rate (7.0), should be rank 1
        assert!((relative[0].win_rate_rank - 1.0).abs() < 0.01);

        // Entry 1 has lowest win rate (5.0), should be rank 3
        assert!((relative[1].win_rate_rank - 3.0).abs() < 0.01);

        // Check course advantage
        assert!((relative[0].course_advantage - 0.55).abs() < 0.01); // boat 1
        assert!((relative[1].course_advantage - 0.14).abs() < 0.01); // boat 2
        assert!((relative[2].course_advantage - 0.12).abs() < 0.01); // boat 3

        // Check win rate diff from avg
        let avg = (7.0 + 5.0 + 6.0) / 3.0;
        assert!((relative[0].win_rate_diff_from_avg - (7.0 - avg)).abs() < 0.01);
    }

    #[test]
    fn test_create_race_features() {
        let entries = create_test_entries();
        let features = FeatureEngineering::create_race_features(&entries);

        assert_eq!(features.len(), 3);

        // Check first entry
        assert_eq!(features[0].boat_no, 1);
        assert!((features[0].base.national_win_rate - 7.0).abs() < 0.01);
        assert!((features[0].relative.win_rate_rank - 1.0).abs() < 0.01);
        assert!(features[0].historical.is_none());
    }

    #[test]
    fn test_racer_features_to_vec() {
        let entries = create_test_entries();
        let features = FeatureEngineering::create_race_features(&entries);

        let vec = features[0].to_vec();

        // Without historical: 10 base + 5 relative + 3 exhibition + 2 context + 6 interaction = 26
        assert_eq!(vec.len(), 26);

        // Check some values (order: base + relative + exhibition + context + interaction)
        assert!((vec[0] - 7.0).abs() < 0.01); // national_win_rate
        assert!((vec[6] - 4.0).abs() < 0.01); // class_encoded (A1)
        assert!((vec[10] - 1.0).abs() < 0.01); // win_rate_rank (first relative feature)
        assert!((vec[15] - 6.80).abs() < 0.01); // exhibition_time (default)
    }

    #[test]
    fn test_racer_features_to_vec_with_historical() {
        let entries = create_test_entries();
        let mut features = FeatureEngineering::create_race_features(&entries);

        // Add historical features
        features[0].historical = Some(HistoricalFeatures {
            recent_win_rate: 0.25,
            recent_in2_rate: 0.45,
            recent_in3_rate: 0.60,
            recent_avg_rank: 2.5,
            recent_avg_st: 0.12,
            recent_race_count: 30.0,
            local_recent_win_rate: 0.30,
            local_race_count: 10.0,
            course_win_rate: 0.35,
            course_in2_rate: 0.50,
            st_consistency: 0.05,
            flying_start_rate: 0.0,
            late_start_rate: 0.1,
            avg_course_diff: 0.0,
            inside_take_rate: 0.0,
            weighted_recent_win: 0.25,
        });

        let vec = features[0].to_vec();

        // With historical: 10 base + 16 historical + 5 relative + 3 exhibition + 2 context + 6 interaction = 42
        assert_eq!(vec.len(), 42);

        // Check order: base (0-9), historical (10-25), relative (26-30), exhibition (31-33), context (34-35), interaction (36-41)
        assert!((vec[0] - 7.0).abs() < 0.01); // national_win_rate (base)
        assert!((vec[10] - 0.25).abs() < 0.01); // recent_win_rate (first historical)
        assert!((vec[26] - 1.0).abs() < 0.01); // win_rate_rank (first relative)
        assert!((vec[31] - 6.80).abs() < 0.01); // exhibition_time (default)
    }

    #[test]
    fn test_feature_names() {
        let base_names = get_base_feature_names();
        assert_eq!(base_names.len(), 26); // 10 base + 5 relative + 3 exhibition + 2 context + 6 interaction

        let all_names = get_all_feature_names();
        assert_eq!(all_names.len(), 42); // 10 base + 16 historical + 5 relative + 3 exhibition + 2 context + 6 interaction
    }

    #[test]
    fn test_exhibition_features_with_times() {
        let entries = create_test_entries();
        let times = [6.75, 6.80, 6.85, 6.90, 6.95, 7.00];

        let features =
            FeatureEngineering::create_race_features_with_exhibition(&entries, Some(&times));

        // Boat 1 has fastest time (6.75), should be rank 1
        assert!((features[0].exhibition.exhibition_time - 6.75).abs() < 0.01);
        assert!((features[0].exhibition.exhibition_time_rank - 1.0).abs() < 0.01);

        // Boat 2 has second fastest time (6.80), should be rank 2
        assert!((features[1].exhibition.exhibition_time - 6.80).abs() < 0.01);
        assert!((features[1].exhibition.exhibition_time_rank - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_empty_entries() {
        let entries: Vec<ProgramEntry> = Vec::new();
        let relative = FeatureEngineering::create_relative_features(&entries);
        assert!(relative.is_empty());

        let features = FeatureEngineering::create_race_features(&entries);
        assert!(features.is_empty());
    }
}
