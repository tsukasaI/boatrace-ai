use serde::{Deserialize, Serialize};

/// Racer entry data from program
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RacerEntry {
    pub boat_no: u8,
    pub racer_id: u32,
    pub racer_name: String,
    pub age: u8,
    pub weight: u8,
    pub racer_class: String,
    pub branch: String,
    pub national_win_rate: f64,
    pub national_in2_rate: f64,
    pub local_win_rate: f64,
    pub local_in2_rate: f64,
    pub motor_no: u32,
    pub motor_in2_rate: f64,
    pub boat_no_equip: u32,
    pub boat_in2_rate: f64,
}

/// Race prediction request
#[derive(Debug, Serialize, Deserialize)]
pub struct PredictRequest {
    pub date: u32,
    pub stadium_code: u8,
    pub race_no: u8,
    pub entries: Vec<RacerEntry>,
    #[serde(default)]
    pub odds: Option<Vec<ExactaOdds>>,
}

/// Exacta odds data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExactaOdds {
    pub first: u8,
    pub second: u8,
    pub odds: f64,
}

/// Trifecta odds data (3連単)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrifectaOdds {
    pub first: u8,
    pub second: u8,
    pub third: u8,
    pub odds: f64,
}

/// Position probability for a boat
#[derive(Debug, Serialize, Deserialize)]
pub struct PositionProb {
    pub boat_no: u8,
    pub probs: [f64; 6], // P(1st), P(2nd), ..., P(6th)
}

/// Exacta prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExactaPrediction {
    pub first: u8,
    pub second: u8,
    pub probability: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub odds: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expected_value: Option<f64>,
    pub is_value_bet: bool,
}

/// Trifecta prediction (3連単)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrifectaPrediction {
    pub first: u8,
    pub second: u8,
    pub third: u8,
    pub probability: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub odds: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expected_value: Option<f64>,
    pub is_value_bet: bool,
}

/// Race prediction response
#[derive(Debug, Serialize, Deserialize)]
pub struct PredictResponse {
    pub date: u32,
    pub stadium_code: u8,
    pub race_no: u8,
    pub position_probs: Vec<PositionProb>,
    pub exacta_predictions: Vec<ExactaPrediction>,
    pub value_bets: Vec<ExactaPrediction>,
}

/// Health check response
#[derive(Debug, Serialize, Deserialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
    pub model_loaded: bool,
}

/// Error response
#[derive(Debug, Serialize, Deserialize)]
pub struct ErrorResponse {
    pub error: String,
    pub message: String,
}
