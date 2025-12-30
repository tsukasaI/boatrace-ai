use actix_web::{middleware, web, App, HttpServer};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tracing::{info, warn, Level};
use tracing_subscriber::FmtSubscriber;

mod error;
mod handlers;
mod models;
mod predictor;

use handlers::{health, predict};
use predictor::{FallbackPredictor, Predictor};

/// Application state shared across handlers
pub struct AppState {
    pub predictor: Option<Mutex<Predictor>>,
    pub fallback_predictor: FallbackPredictor,
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    // Initialize logging
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber).expect("Failed to set subscriber");

    let host = std::env::var("HOST").unwrap_or_else(|_| "127.0.0.1".to_string());
    let port = std::env::var("PORT").unwrap_or_else(|_| "8080".to_string());
    let addr = format!("{}:{}", host, port);

    // Load ONNX models
    let model_dir = std::env::var("MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("../models/onnx"));

    info!("Loading ONNX models from {:?}", model_dir);

    let predictor = match Predictor::new(&model_dir) {
        Ok(p) => {
            info!("ONNX models loaded successfully");
            Some(Mutex::new(p))
        }
        Err(e) => {
            warn!("Failed to load ONNX models: {}. Using fallback predictor.", e);
            None
        }
    };

    let app_state = Arc::new(AppState {
        predictor,
        fallback_predictor: FallbackPredictor::new(),
    });

    info!("Starting Boatrace API server at http://{}", addr);

    HttpServer::new(move || {
        App::new()
            .app_data(web::Data::new(app_state.clone()))
            .wrap(middleware::Logger::default())
            .route("/health", web::get().to(health::health_check))
            .route("/predict", web::post().to(predict::predict_race))
            .route("/predict/exacta", web::post().to(predict::predict_exacta))
    })
    .bind(&addr)?
    .run()
    .await
}
