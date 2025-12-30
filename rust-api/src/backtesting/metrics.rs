//! Backtest Metrics
//!
//! Calculate metrics such as ROI, hit rate, drawdown, etc.

use super::simulator::BetRecord;
use serde::{Deserialize, Serialize};

/// Backtest evaluation metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestMetrics {
    // Basic metrics
    pub total_bets: usize,
    pub winning_bets: usize,
    pub hit_rate: f64,
    pub roi: f64,

    // Expected value related
    pub avg_ev: f64,
    pub avg_odds: f64,
    pub avg_probability: f64,

    // Risk metrics
    pub profit_factor: f64,
    pub max_drawdown: i64,
    pub max_drawdown_pct: f64,

    // Win/Loss
    pub gross_profit: i64,
    pub gross_loss: i64,
    pub net_profit: i64,
}

impl Default for BacktestMetrics {
    fn default() -> Self {
        Self {
            total_bets: 0,
            winning_bets: 0,
            hit_rate: 0.0,
            roi: 0.0,
            avg_ev: 0.0,
            avg_odds: 0.0,
            avg_probability: 0.0,
            profit_factor: 0.0,
            max_drawdown: 0,
            max_drawdown_pct: 0.0,
            gross_profit: 0,
            gross_loss: 0,
            net_profit: 0,
        }
    }
}

/// Calculate metrics from bet records
pub fn calculate_metrics(bets: &[BetRecord], total_stake: i64) -> BacktestMetrics {
    if bets.is_empty() {
        return BacktestMetrics::default();
    }

    // Basic metrics
    let total_bets = bets.len();
    let winning_bets = bets.iter().filter(|b| b.won).count();
    let hit_rate = winning_bets as f64 / total_bets as f64;

    // Expected value related
    let avg_ev: f64 = bets.iter().map(|b| b.expected_value).sum::<f64>() / total_bets as f64;
    let avg_odds: f64 = bets.iter().map(|b| b.odds).sum::<f64>() / total_bets as f64;
    let avg_probability: f64 = bets.iter().map(|b| b.probability).sum::<f64>() / total_bets as f64;

    // Profit/Loss calculation
    let profits: Vec<i64> = bets.iter().map(|b| b.profit).collect();
    let gross_profit: i64 = profits.iter().filter(|&&p| p > 0).sum();
    let gross_loss: i64 = profits.iter().filter(|&&p| p < 0).map(|p| p.abs()).sum();
    let net_profit: i64 = profits.iter().sum();

    // Profit Factor
    let profit_factor = if gross_loss > 0 {
        gross_profit as f64 / gross_loss as f64
    } else if gross_profit > 0 {
        f64::INFINITY
    } else {
        0.0
    };

    // Drawdown calculation
    let mut cumulative: Vec<i64> = Vec::with_capacity(profits.len());
    let mut sum = 0i64;
    for &p in &profits {
        sum += p;
        cumulative.push(sum);
    }

    let mut peak = i64::MIN;
    let mut max_drawdown = 0i64;
    for &value in &cumulative {
        if value > peak {
            peak = value;
        }
        let drawdown = peak - value;
        if drawdown > max_drawdown {
            max_drawdown = drawdown;
        }
    }

    // Drawdown percentage
    let max_drawdown_pct = if total_stake > 0 {
        max_drawdown as f64 / total_stake as f64
    } else {
        0.0
    };

    // ROI
    let roi = if total_stake > 0 {
        net_profit as f64 / total_stake as f64
    } else {
        0.0
    };

    BacktestMetrics {
        total_bets,
        winning_bets,
        hit_rate,
        roi,
        avg_ev,
        avg_odds,
        avg_probability,
        profit_factor,
        max_drawdown,
        max_drawdown_pct,
        gross_profit,
        gross_loss,
        net_profit,
    }
}

/// Calculate Sharpe ratio from bet records
pub fn calculate_sharpe_ratio(bets: &[BetRecord], risk_free_rate: f64) -> f64 {
    if bets.is_empty() {
        return 0.0;
    }

    let returns: Vec<f64> = bets
        .iter()
        .map(|b| b.profit as f64 / b.stake as f64)
        .collect();

    let mean_return: f64 = returns.iter().sum::<f64>() / returns.len() as f64;

    let variance: f64 = returns
        .iter()
        .map(|r| (r - mean_return).powi(2))
        .sum::<f64>()
        / returns.len() as f64;

    let std_return = variance.sqrt();

    if std_return == 0.0 {
        return 0.0;
    }

    (mean_return - risk_free_rate) / std_return
}

/// Analysis results by dimension
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DimensionAnalysis {
    pub key: String,
    pub bets: usize,
    pub wins: usize,
    pub hit_rate: f64,
    pub stake: i64,
    pub profit: i64,
    pub roi: f64,
}

/// Analyze bet results by stadium
pub fn analyze_by_stadium(bets: &[BetRecord]) -> Vec<DimensionAnalysis> {
    use std::collections::HashMap;

    let mut grouped: HashMap<u8, Vec<&BetRecord>> = HashMap::new();
    for bet in bets {
        grouped.entry(bet.stadium_code).or_default().push(bet);
    }

    let mut results: Vec<DimensionAnalysis> = grouped
        .iter()
        .map(|(stadium, group)| {
            let total = group.len();
            let wins = group.iter().filter(|b| b.won).count();
            let stake: i64 = group.iter().map(|b| b.stake).sum();
            let profit: i64 = group.iter().map(|b| b.profit).sum();

            DimensionAnalysis {
                key: stadium.to_string(),
                bets: total,
                wins,
                hit_rate: if total > 0 {
                    wins as f64 / total as f64
                } else {
                    0.0
                },
                stake,
                profit,
                roi: if stake > 0 {
                    profit as f64 / stake as f64
                } else {
                    0.0
                },
            }
        })
        .collect();

    results.sort_by(|a, b| a.key.cmp(&b.key));
    results
}

/// Analyze bet results by odds range
pub fn analyze_by_odds_range(bets: &[BetRecord]) -> Vec<DimensionAnalysis> {
    use std::collections::HashMap;

    let mut grouped: HashMap<&str, Vec<&BetRecord>> = HashMap::new();
    for bet in bets {
        let key = if bet.odds < 5.0 {
            "low (<5)"
        } else if bet.odds < 20.0 {
            "mid (5-20)"
        } else {
            "high (>20)"
        };
        grouped.entry(key).or_default().push(bet);
    }

    let mut results: Vec<DimensionAnalysis> = grouped
        .iter()
        .map(|(key, group)| {
            let total = group.len();
            let wins = group.iter().filter(|b| b.won).count();
            let stake: i64 = group.iter().map(|b| b.stake).sum();
            let profit: i64 = group.iter().map(|b| b.profit).sum();

            DimensionAnalysis {
                key: key.to_string(),
                bets: total,
                wins,
                hit_rate: if total > 0 {
                    wins as f64 / total as f64
                } else {
                    0.0
                },
                stake,
                profit,
                roi: if stake > 0 {
                    profit as f64 / stake as f64
                } else {
                    0.0
                },
            }
        })
        .collect();

    results.sort_by(|a, b| a.key.cmp(&b.key));
    results
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_bets() -> Vec<BetRecord> {
        vec![
            BetRecord {
                date: 20240115,
                stadium_code: 23,
                race_no: 1,
                first: 1,
                second: 2,
                probability: 0.10,
                odds: 8.0,
                expected_value: 0.80,
                stake: 100,
                actual_first: 1,
                actual_second: 2,
                won: true,
                profit: 700, // 800 - 100
            },
            BetRecord {
                date: 20240115,
                stadium_code: 23,
                race_no: 2,
                first: 1,
                second: 3,
                probability: 0.08,
                odds: 15.0,
                expected_value: 1.20,
                stake: 100,
                actual_first: 2,
                actual_second: 1,
                won: false,
                profit: -100,
            },
            BetRecord {
                date: 20240115,
                stadium_code: 24,
                race_no: 1,
                first: 2,
                second: 1,
                probability: 0.12,
                odds: 10.0,
                expected_value: 1.20,
                stake: 100,
                actual_first: 2,
                actual_second: 1,
                won: true,
                profit: 900, // 1000 - 100
            },
        ]
    }

    #[test]
    fn test_calculate_metrics() {
        let bets = create_test_bets();
        let metrics = calculate_metrics(&bets, 300);

        assert_eq!(metrics.total_bets, 3);
        assert_eq!(metrics.winning_bets, 2);
        assert!((metrics.hit_rate - 0.6667).abs() < 0.01);
        assert_eq!(metrics.gross_profit, 1600); // 700 + 900
        assert_eq!(metrics.gross_loss, 100);
        assert_eq!(metrics.net_profit, 1500); // 700 - 100 + 900
    }

    #[test]
    fn test_calculate_metrics_empty() {
        let bets: Vec<BetRecord> = Vec::new();
        let metrics = calculate_metrics(&bets, 0);

        assert_eq!(metrics.total_bets, 0);
        assert_eq!(metrics.winning_bets, 0);
        assert_eq!(metrics.hit_rate, 0.0);
    }

    #[test]
    fn test_calculate_sharpe_ratio() {
        let bets = create_test_bets();
        let sharpe = calculate_sharpe_ratio(&bets, 0.0);

        // Returns: 7.0, -1.0, 9.0 -> mean = 5.0, std ≈ 4.32
        // Sharpe ≈ 5.0 / 4.32 ≈ 1.16
        assert!(sharpe > 0.0);
    }

    #[test]
    fn test_analyze_by_stadium() {
        let bets = create_test_bets();
        let analysis = analyze_by_stadium(&bets);

        assert_eq!(analysis.len(), 2); // stadiums 23 and 24

        let stadium_23 = analysis.iter().find(|a| a.key == "23").unwrap();
        assert_eq!(stadium_23.bets, 2);
        assert_eq!(stadium_23.wins, 1);

        let stadium_24 = analysis.iter().find(|a| a.key == "24").unwrap();
        assert_eq!(stadium_24.bets, 1);
        assert_eq!(stadium_24.wins, 1);
    }

    #[test]
    fn test_analyze_by_odds_range() {
        let bets = create_test_bets();
        let analysis = analyze_by_odds_range(&bets);

        // odds: 8.0 (mid), 15.0 (mid), 10.0 (mid)
        assert_eq!(analysis.len(), 1);
        assert_eq!(analysis[0].key, "mid (5-20)");
        assert_eq!(analysis[0].bets, 3);
    }

    #[test]
    fn test_max_drawdown() {
        // Create a sequence with drawdown
        let bets = vec![
            BetRecord {
                date: 20240115,
                stadium_code: 23,
                race_no: 1,
                first: 1,
                second: 2,
                probability: 0.10,
                odds: 10.0,
                expected_value: 1.0,
                stake: 100,
                actual_first: 1,
                actual_second: 2,
                won: true,
                profit: 900,
            },
            BetRecord {
                date: 20240115,
                stadium_code: 23,
                race_no: 2,
                first: 1,
                second: 2,
                probability: 0.10,
                odds: 10.0,
                expected_value: 1.0,
                stake: 100,
                actual_first: 2,
                actual_second: 1,
                won: false,
                profit: -100,
            },
            BetRecord {
                date: 20240115,
                stadium_code: 23,
                race_no: 3,
                first: 1,
                second: 2,
                probability: 0.10,
                odds: 10.0,
                expected_value: 1.0,
                stake: 100,
                actual_first: 2,
                actual_second: 1,
                won: false,
                profit: -100,
            },
        ];

        let metrics = calculate_metrics(&bets, 300);

        // Cumulative: 900, 800, 700
        // Peak: 900, 900, 900
        // Drawdown: 0, 100, 200
        // Max drawdown: 200
        assert_eq!(metrics.max_drawdown, 200);
    }
}
