use std::io::{self, Stdout};
use std::path::PathBuf;
use std::time::Instant;

use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use ratatui::backend::CrosstermBackend;
use ratatui::Terminal;

use crate::error::SomaError;
use crate::eval::EvalReport;
use crate::io::{load_csv, split_train_test};
use crate::models::store::ModelMetadata;
use crate::models::{Algorithm, ModelStore};

use super::event::{next_event, AppEvent};
use super::ui;

pub fn run_tui() -> anyhow::Result<()> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;
    terminal.clear()?;

    let result = run_app(&mut terminal);

    // Always restore the terminal, even if run_app blew up.
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    result
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Screen {
    MainMenu,
    TrainForm,
    PredictForm,
    EvaluateForm,
    InspectForm,
    AlgorithmsList,
    Results,
    Help,
}

pub static MENU_ITEMS: &[&str] = &[
    "Train a model",
    "Make predictions",
    "Evaluate a model",
    "Inspect a model",
    "List algorithms",
    "Help",
    "Quit",
];

#[derive(Debug, Clone)]
pub struct FormField {
    pub label: String,
    pub value: String,
    pub placeholder: String,
    pub required: bool,
    pub choices: Option<Vec<String>>,
    pub choice_descriptions: Option<Vec<String>>,
    pub choice_idx: usize,
    pub cursor: usize,
}

impl FormField {
    fn text(label: &str, placeholder: &str, required: bool) -> Self {
        Self {
            label: label.to_string(),
            value: String::new(),
            placeholder: placeholder.to_string(),
            required,
            choices: None,
            choice_descriptions: None,
            choice_idx: 0,
            cursor: 0,
        }
    }

    fn text_with_default(label: &str, placeholder: &str, default: &str, required: bool) -> Self {
        let cursor = default.len();
        Self {
            label: label.to_string(),
            value: default.to_string(),
            placeholder: placeholder.to_string(),
            required,
            choices: None,
            choice_descriptions: None,
            choice_idx: 0,
            cursor,
        }
    }

    fn select(label: &str, choices: Vec<String>, descriptions: Vec<String>) -> Self {
        Self {
            label: label.to_string(),
            value: choices.first().cloned().unwrap_or_default(),
            placeholder: String::new(),
            required: true,
            choices: Some(choices),
            choice_descriptions: Some(descriptions),
            choice_idx: 0,
            cursor: 0,
        }
    }

    pub fn selected_description(&self) -> Option<&str> {
        self.choice_descriptions
            .as_ref()
            .and_then(|descs| descs.get(self.choice_idx))
            .map(|s| s.as_str())
    }

    pub fn choice_count(&self) -> usize {
        self.choices.as_ref().map_or(0, |c| c.len())
    }

    pub fn effective_value(&self) -> String {
        if let Some(ref choices) = self.choices {
            choices.get(self.choice_idx).cloned().unwrap_or_default()
        } else {
            self.value.trim().to_string()
        }
    }

    pub fn is_filled(&self) -> bool {
        !self.effective_value().is_empty()
    }

    pub fn insert_char(&mut self, c: char) {
        if self.choices.is_some() {
            return;
        }
        let byte_idx = self
            .value
            .char_indices()
            .nth(self.cursor)
            .map(|(i, _)| i)
            .unwrap_or(self.value.len());
        self.value.insert(byte_idx, c);
        self.cursor += 1;
    }

    pub fn delete_char_before(&mut self) {
        if self.choices.is_some() || self.cursor == 0 {
            return;
        }
        self.cursor -= 1;
        let byte_idx = self
            .value
            .char_indices()
            .nth(self.cursor)
            .map(|(i, _)| i)
            .unwrap_or(self.value.len());
        let end_idx = self
            .value
            .char_indices()
            .nth(self.cursor + 1)
            .map(|(i, _)| i)
            .unwrap_or(self.value.len());
        self.value.drain(byte_idx..end_idx);
    }

    pub fn delete_char_at(&mut self) {
        if self.choices.is_some() {
            return;
        }
        let char_count = self.value.chars().count();
        if self.cursor >= char_count {
            return;
        }
        let byte_idx = self
            .value
            .char_indices()
            .nth(self.cursor)
            .map(|(i, _)| i)
            .unwrap_or(self.value.len());
        let end_idx = self
            .value
            .char_indices()
            .nth(self.cursor + 1)
            .map(|(i, _)| i)
            .unwrap_or(self.value.len());
        self.value.drain(byte_idx..end_idx);
    }

    pub fn cursor_left(&mut self) {
        if let Some(ref choices) = self.choices {
            if self.choice_idx > 0 {
                self.choice_idx -= 1;
                self.value = choices[self.choice_idx].clone();
            }
        } else if self.cursor > 0 {
            self.cursor -= 1;
        }
    }

    pub fn cursor_right(&mut self) {
        if let Some(ref choices) = self.choices {
            if self.choice_idx + 1 < choices.len() {
                self.choice_idx += 1;
                self.value = choices[self.choice_idx].clone();
            }
        } else {
            let char_count = self.value.chars().count();
            if self.cursor < char_count {
                self.cursor += 1;
            }
        }
    }

    pub fn cursor_home(&mut self) {
        if self.choices.is_some() {
            self.choice_idx = 0;
            if let Some(ref choices) = self.choices {
                self.value = choices[0].clone();
            }
        } else {
            self.cursor = 0;
        }
    }

    pub fn cursor_end(&mut self) {
        if let Some(ref choices) = self.choices {
            self.choice_idx = choices.len().saturating_sub(1);
            self.value = choices[self.choice_idx].clone();
        } else {
            self.cursor = self.value.chars().count();
        }
    }
}

pub struct App {
    pub should_quit: bool,
    pub screen: Screen,
    pub menu_index: usize,
    pub form_fields: Vec<FormField>,
    pub form_focus: usize,
    pub submit_focused: bool,
    pub result_lines: Vec<String>,
    pub result_title: String,
    pub result_is_error: bool,
    pub result_scroll: u16,
    pub status_message: String,
    pub algo_scroll: usize,
    pub help_scroll: u16,
}

impl App {
    pub fn new() -> Self {
        Self {
            should_quit: false,
            screen: Screen::MainMenu,
            menu_index: 0,
            form_fields: Vec::new(),
            form_focus: 0,
            submit_focused: false,
            result_lines: Vec::new(),
            result_title: String::new(),
            result_is_error: false,
            result_scroll: 0,
            status_message: "Welcome to soma! Use ↑↓ to navigate, Enter to select.".into(),
            algo_scroll: 0,
            help_scroll: 0,
        }
    }

    fn build_train_form(&mut self) {
        let algo_names: Vec<String> = Algorithm::all()
            .iter()
            .map(|a| algorithm_cli_name(*a))
            .collect();
        let algo_descs: Vec<String> = Algorithm::all()
            .iter()
            .map(|a| a.description().to_string())
            .collect();

        self.form_fields = vec![
            FormField::text("Data file", "path/to/data.csv", true),
            FormField::text("Target column", "column name or index", true),
            FormField::select("Algorithm", algo_names, algo_descs),
            FormField::text_with_default("Output file", "path/to/model.soma", "model.soma", true),
            FormField::text_with_default("Test size", "0.0–1.0", "0.2", true),
        ];
        self.form_focus = 0;
        self.submit_focused = false;
    }

    fn build_predict_form(&mut self) {
        self.form_fields = vec![
            FormField::text("Model file", "path/to/model.soma", true),
            FormField::text("Data file", "path/to/data.csv", true),
            FormField::text("Target column (optional)", "column to exclude", false),
            FormField::text("Output file (optional)", "path/to/predictions.csv", false),
        ];
        self.form_focus = 0;
        self.submit_focused = false;
    }

    fn build_evaluate_form(&mut self) {
        let algo_names: Vec<String> = Algorithm::all()
            .iter()
            .map(|a| algorithm_cli_name(*a))
            .collect();
        let algo_descs: Vec<String> = Algorithm::all()
            .iter()
            .map(|a| a.description().to_string())
            .collect();

        self.form_fields = vec![
            FormField::text("Data file", "path/to/data.csv", true),
            FormField::text("Target column", "column name or index", true),
            FormField::select("Algorithm", algo_names, algo_descs),
            FormField::text_with_default("Test size", "0.0–1.0", "0.2", true),
        ];
        self.form_focus = 0;
        self.submit_focused = false;
    }

    fn build_inspect_form(&mut self) {
        self.form_fields = vec![FormField::text("Model file", "path/to/model.soma", true)];
        self.form_focus = 0;
        self.submit_focused = false;
    }

    fn goto(&mut self, screen: Screen) {
        match screen {
            Screen::TrainForm => self.build_train_form(),
            Screen::PredictForm => self.build_predict_form(),
            Screen::EvaluateForm => self.build_evaluate_form(),
            Screen::InspectForm => self.build_inspect_form(),
            Screen::AlgorithmsList => self.algo_scroll = 0,
            Screen::Results => self.result_scroll = 0,
            Screen::Help => self.help_scroll = 0,
            Screen::MainMenu => {
                self.status_message = "Use ↑↓ to navigate, Enter to select, q to quit.".into();
            }
        }
        self.screen = screen;
    }

    fn back_to_menu(&mut self) {
        self.goto(Screen::MainMenu);
    }

    pub fn handle_event(&mut self, event: AppEvent) {
        if event == AppEvent::CtrlC {
            self.should_quit = true;
            return;
        }

        match self.screen {
            Screen::MainMenu => self.handle_menu_event(event),
            Screen::TrainForm
            | Screen::PredictForm
            | Screen::EvaluateForm
            | Screen::InspectForm => self.handle_form_event(event),
            Screen::AlgorithmsList => self.handle_algorithms_event(event),
            Screen::Results => self.handle_results_event(event),
            Screen::Help => self.handle_help_event(event),
        }
    }

    fn handle_menu_event(&mut self, event: AppEvent) {
        match event {
            AppEvent::Up => {
                if self.menu_index > 0 {
                    self.menu_index -= 1;
                } else {
                    self.menu_index = MENU_ITEMS.len() - 1;
                }
            }
            AppEvent::Down => {
                if self.menu_index + 1 < MENU_ITEMS.len() {
                    self.menu_index += 1;
                } else {
                    self.menu_index = 0;
                }
            }
            AppEvent::Enter => self.activate_menu_item(),
            AppEvent::Char('q') | AppEvent::Char('Q') => {
                self.should_quit = true;
            }
            AppEvent::Char('1') => {
                self.menu_index = 0;
                self.activate_menu_item();
            }
            AppEvent::Char('2') => {
                self.menu_index = 1;
                self.activate_menu_item();
            }
            AppEvent::Char('3') => {
                self.menu_index = 2;
                self.activate_menu_item();
            }
            AppEvent::Char('4') => {
                self.menu_index = 3;
                self.activate_menu_item();
            }
            AppEvent::Char('5') => {
                self.menu_index = 4;
                self.activate_menu_item();
            }
            AppEvent::Char('6') | AppEvent::F1 => {
                self.menu_index = 5;
                self.activate_menu_item();
            }
            _ => {}
        }
    }

    fn activate_menu_item(&mut self) {
        match self.menu_index {
            0 => self.goto(Screen::TrainForm),
            1 => self.goto(Screen::PredictForm),
            2 => self.goto(Screen::EvaluateForm),
            3 => self.goto(Screen::InspectForm),
            4 => self.goto(Screen::AlgorithmsList),
            5 => self.goto(Screen::Help),
            6 => self.should_quit = true,
            _ => {}
        }
    }

    fn handle_form_event(&mut self, event: AppEvent) {
        match event {
            AppEvent::Escape => self.back_to_menu(),
            AppEvent::Tab | AppEvent::Down => self.focus_next_field(),
            AppEvent::BackTab | AppEvent::Up => self.focus_prev_field(),
            AppEvent::Enter => {
                if self.submit_focused {
                    self.submit_form();
                } else {
                    self.focus_next_field();
                }
            }
            AppEvent::Left => {
                if !self.submit_focused {
                    if let Some(field) = self.form_fields.get_mut(self.form_focus) {
                        field.cursor_left();
                    }
                }
            }
            AppEvent::Right => {
                if !self.submit_focused {
                    if let Some(field) = self.form_fields.get_mut(self.form_focus) {
                        field.cursor_right();
                    }
                }
            }
            AppEvent::Home => {
                if !self.submit_focused {
                    if let Some(field) = self.form_fields.get_mut(self.form_focus) {
                        field.cursor_home();
                    }
                }
            }
            AppEvent::End => {
                if !self.submit_focused {
                    if let Some(field) = self.form_fields.get_mut(self.form_focus) {
                        field.cursor_end();
                    }
                }
            }
            AppEvent::Char(c) => {
                if !self.submit_focused {
                    if let Some(field) = self.form_fields.get_mut(self.form_focus) {
                        field.insert_char(c);
                    }
                }
            }
            AppEvent::Backspace => {
                if !self.submit_focused {
                    if let Some(field) = self.form_fields.get_mut(self.form_focus) {
                        field.delete_char_before();
                    }
                }
            }
            AppEvent::Delete => {
                if !self.submit_focused {
                    if let Some(field) = self.form_fields.get_mut(self.form_focus) {
                        field.delete_char_at();
                    }
                }
            }
            _ => {}
        }
    }

    fn focus_next_field(&mut self) {
        if self.submit_focused {
            // Wrap around to first field.
            self.submit_focused = false;
            self.form_focus = 0;
        } else if self.form_focus + 1 < self.form_fields.len() {
            self.form_focus += 1;
        } else {
            self.submit_focused = true;
        }
    }

    fn focus_prev_field(&mut self) {
        if self.submit_focused {
            self.submit_focused = false;
            self.form_focus = self.form_fields.len().saturating_sub(1);
        } else if self.form_focus > 0 {
            self.form_focus -= 1;
        } else {
            self.submit_focused = true;
        }
    }

    fn submit_form(&mut self) {
        for field in &self.form_fields {
            if field.required && !field.is_filled() {
                self.status_message = format!("'{}' is required.", field.label);
                return;
            }
        }

        let screen = self.screen.clone();
        match screen {
            Screen::TrainForm => self.execute_train(),
            Screen::PredictForm => self.execute_predict(),
            Screen::EvaluateForm => self.execute_evaluate(),
            Screen::InspectForm => self.execute_inspect(),
            _ => {}
        }
    }

    fn handle_algorithms_event(&mut self, event: AppEvent) {
        let total = Algorithm::all().len();
        match event {
            AppEvent::Escape | AppEvent::Char('q') | AppEvent::Char('Q') => self.back_to_menu(),
            AppEvent::Up => {
                if self.algo_scroll > 0 {
                    self.algo_scroll -= 1;
                }
            }
            AppEvent::Down => {
                if self.algo_scroll + 1 < total {
                    self.algo_scroll += 1;
                }
            }
            AppEvent::Home => self.algo_scroll = 0,
            AppEvent::End => self.algo_scroll = total.saturating_sub(1),
            _ => {}
        }
    }

    fn handle_results_event(&mut self, event: AppEvent) {
        match event {
            AppEvent::Escape | AppEvent::Char('q') | AppEvent::Char('Q') | AppEvent::Enter => {
                self.back_to_menu();
            }
            AppEvent::Up => {
                self.result_scroll = self.result_scroll.saturating_sub(1);
            }
            AppEvent::Down => {
                self.result_scroll = self.result_scroll.saturating_add(1);
            }
            AppEvent::Home => self.result_scroll = 0,
            AppEvent::End => {
                self.result_scroll = self.result_lines.len().saturating_sub(1) as u16;
            }
            _ => {}
        }
    }

    fn handle_help_event(&mut self, event: AppEvent) {
        match event {
            AppEvent::Escape
            | AppEvent::Char('q')
            | AppEvent::Char('Q')
            | AppEvent::Enter
            | AppEvent::F1 => {
                self.back_to_menu();
            }
            AppEvent::Up => {
                self.help_scroll = self.help_scroll.saturating_sub(1);
            }
            AppEvent::Down => {
                self.help_scroll = self.help_scroll.saturating_add(1);
            }
            _ => {}
        }
    }

    fn execute_train(&mut self) {
        let data_path = PathBuf::from(self.form_fields[0].effective_value());
        let target = self.form_fields[1].effective_value();
        let algo_name = self.form_fields[2].effective_value();
        let output_path = PathBuf::from(self.form_fields[3].effective_value());
        let test_size_str = self.form_fields[4].effective_value();

        let algorithm = match parse_algorithm(&algo_name) {
            Ok(a) => a,
            Err(e) => return self.show_error("Train Error", &e),
        };

        let test_size: f64 = match test_size_str.parse() {
            Ok(v) => v,
            Err(_) => {
                return self.show_error(
                    "Train Error",
                    &format!("Invalid test size: '{test_size_str}'. Must be a number 0.0–1.0."),
                );
            }
        };

        if !(0.0..1.0).contains(&test_size) {
            return self.show_error(
                "Train Error",
                &format!("Test size must be >= 0.0 and < 1.0, got {test_size}."),
            );
        }

        let mut lines: Vec<String> = Vec::new();

        let dataset = match load_csv(&data_path, &target) {
            Ok(ds) => ds,
            Err(e) => return self.show_error("Train Error", &e.to_string()),
        };
        lines.push(format!("Loaded: {dataset}"));

        let (train_x, train_y, eval_split) = if test_size > 0.0 && test_size < 1.0 {
            match split_train_test(&dataset, test_size) {
                Ok(split) => {
                    lines.push(format!(
                        "Split: {} train / {} test ({:.0}%)",
                        split.n_train,
                        split.n_test,
                        test_size * 100.0
                    ));
                    let eval = Some((split.x_test, split.y_test));
                    (split.x_train, split.y_train, eval)
                }
                Err(e) => return self.show_error("Train Error", &e.to_string()),
            }
        } else {
            lines.push(format!(
                "Using all {} samples (no test split)",
                dataset.n_samples
            ));
            (dataset.x.clone(), dataset.y.clone(), None)
        };

        lines.push(String::new());
        lines.push(format!(
            "Training {} on {} samples with {} features...",
            algorithm.description(),
            train_y.len(),
            dataset.n_features,
        ));

        let start = Instant::now();
        let model = match algorithm.train(&train_x, &train_y) {
            Ok(m) => m,
            Err(e) => return self.show_error("Train Error", &e.to_string()),
        };
        let elapsed = start.elapsed();
        lines.push(format!("Training completed in {elapsed:.2?}"));

        if let Some((ref x_test, ref y_test)) = eval_split {
            match model.predict(x_test) {
                Ok(preds) => {
                    let report = EvalReport::compute(algorithm.task(), y_test, &preds);
                    lines.push(String::new());
                    lines.push("── Test Set Evaluation ──".into());
                    for line in report.to_table().lines() {
                        lines.push(line.to_string());
                    }
                }
                Err(e) => {
                    lines.push(format!("Warning: evaluation failed: {e}"));
                }
            }
        }

        let metadata = ModelMetadata {
            algorithm,
            feature_names: dataset.feature_names.clone(),
            target_name: dataset.target_name.clone(),
            n_train_samples: train_y.len(),
            n_features: dataset.n_features,
        };

        let store = ModelStore::new(model, metadata);
        if let Err(e) = store.save(&output_path) {
            return self.show_error("Train Error", &e.to_string());
        }

        lines.push(String::new());
        lines.push(format!("✓ Model saved to '{}'", output_path.display()));

        self.show_results("Training Complete", lines);
    }

    fn execute_predict(&mut self) {
        let model_path = PathBuf::from(self.form_fields[0].effective_value());
        let data_path = PathBuf::from(self.form_fields[1].effective_value());
        let target_col = self.form_fields[2].effective_value();
        let output_file = self.form_fields[3].effective_value();

        let store = match ModelStore::load(&model_path) {
            Ok(s) => s,
            Err(e) => return self.show_error("Predict Error", &e.to_string()),
        };

        let mut lines: Vec<String> = Vec::new();
        lines.push(format!("Model: {}", store.metadata.algorithm.description()));
        lines.push(format!(
            "Features: [{}]",
            store.metadata.feature_names.join(", ")
        ));
        lines.push(format!("Target: {}", store.metadata.target_name));
        lines.push(String::new());

        let target = if target_col.is_empty() {
            &store.metadata.target_name
        } else {
            &target_col
        };

        let (x, n_features) = match load_csv(&data_path, target) {
            Ok(ds) => (ds.x, ds.n_features),
            Err(SomaError::ColumnNotFound(_)) => {
                match load_all_columns(&data_path, store.metadata.n_features) {
                    Ok(result) => result,
                    Err(e) => return self.show_error("Predict Error", &e.to_string()),
                }
            }
            Err(e) => return self.show_error("Predict Error", &e.to_string()),
        };

        if let Err(e) = store.validate_features(n_features) {
            return self.show_error("Predict Error", &e.to_string());
        }

        use smartcore::linalg::basic::arrays::Array;
        let (n_rows, _) = x.shape();
        lines.push(format!("Predicting on {n_rows} samples..."));

        let start = Instant::now();
        let predictions = match store.model.predict(&x) {
            Ok(p) => p,
            Err(e) => return self.show_error("Predict Error", &e.to_string()),
        };
        let elapsed = start.elapsed();
        lines.push(format!("Prediction completed in {elapsed:.2?}"));
        lines.push(String::new());

        if !output_file.is_empty() {
            let out_path = PathBuf::from(&output_file);
            match write_predictions_to_file(&out_path, &predictions) {
                Ok(()) => {
                    lines.push(format!(
                        "✓ {} predictions written to '{}'",
                        predictions.len(),
                        out_path.display()
                    ));
                }
                Err(e) => return self.show_error("Predict Error", &e.to_string()),
            }
        } else {
            lines.push("── Predictions ──".into());
            let max_display = 100;
            for (i, pred) in predictions.iter().enumerate() {
                if i >= max_display {
                    lines.push(format!(
                        "... and {} more (use output file to see all)",
                        predictions.len() - max_display
                    ));
                    break;
                }
                lines.push(format!("  [{i:>4}] {pred}"));
            }
        }

        self.show_results("Predictions", lines);
    }

    fn execute_evaluate(&mut self) {
        let data_path = PathBuf::from(self.form_fields[0].effective_value());
        let target = self.form_fields[1].effective_value();
        let algo_name = self.form_fields[2].effective_value();
        let test_size_str = self.form_fields[3].effective_value();

        let algorithm = match parse_algorithm(&algo_name) {
            Ok(a) => a,
            Err(e) => return self.show_error("Evaluate Error", &e),
        };

        let test_size: f64 = match test_size_str.parse() {
            Ok(v) => v,
            Err(_) => {
                return self.show_error(
                    "Evaluate Error",
                    &format!("Invalid test size: '{test_size_str}'."),
                );
            }
        };

        if test_size <= 0.0 || test_size >= 1.0 {
            return self.show_error(
                "Evaluate Error",
                &format!("Test size must be between 0 and 1 exclusive, got {test_size}."),
            );
        }

        let mut lines: Vec<String> = Vec::new();

        let dataset = match load_csv(&data_path, &target) {
            Ok(ds) => ds,
            Err(e) => return self.show_error("Evaluate Error", &e.to_string()),
        };
        lines.push(format!("Loaded: {dataset}"));

        let split = match split_train_test(&dataset, test_size) {
            Ok(s) => s,
            Err(e) => return self.show_error("Evaluate Error", &e.to_string()),
        };
        lines.push(format!(
            "Split: {} train / {} test ({:.0}%)",
            split.n_train,
            split.n_test,
            test_size * 100.0
        ));
        lines.push(String::new());
        lines.push(format!("Training {}...", algorithm.description()));

        let start = Instant::now();
        let model = match algorithm.train(&split.x_train, &split.y_train) {
            Ok(m) => m,
            Err(e) => return self.show_error("Evaluate Error", &e.to_string()),
        };
        let train_elapsed = start.elapsed();
        lines.push(format!("Training completed in {train_elapsed:.2?}"));

        let start = Instant::now();
        let train_preds = match model.predict(&split.x_train) {
            Ok(p) => p,
            Err(e) => return self.show_error("Evaluate Error", &e.to_string()),
        };
        let test_preds = match model.predict(&split.x_test) {
            Ok(p) => p,
            Err(e) => return self.show_error("Evaluate Error", &e.to_string()),
        };
        let pred_elapsed = start.elapsed();
        lines.push(format!("Prediction completed in {pred_elapsed:.2?}"));

        let task = algorithm.task();
        let train_report = EvalReport::compute(task, &split.y_train, &train_preds);
        let test_report = EvalReport::compute(task, &split.y_test, &test_preds);

        lines.push(String::new());
        lines.push("── Training Set ──".into());
        for line in train_report.to_table().lines() {
            lines.push(line.to_string());
        }

        lines.push(String::new());
        lines.push("── Test Set ──".into());
        for line in test_report.to_table().lines() {
            lines.push(line.to_string());
        }

        self.show_results("Evaluation Results", lines);
    }

    fn execute_inspect(&mut self) {
        let model_path = PathBuf::from(self.form_fields[0].effective_value());

        let store = match ModelStore::load(&model_path) {
            Ok(s) => s,
            Err(e) => return self.show_error("Inspect Error", &e.to_string()),
        };

        let meta = &store.metadata;
        let mut lines: Vec<String> = Vec::new();

        lines.push(format!(
            "  Algorithm:        {}",
            meta.algorithm.description()
        ));
        lines.push(format!("  Task:             {}", meta.algorithm.task()));
        lines.push(format!("  Target column:    {}", meta.target_name));
        lines.push(format!("  Features:         {} columns", meta.n_features));
        lines.push(format!(
            "  Feature names:    [{}]",
            meta.feature_names.join(", ")
        ));
        lines.push(format!("  Training samples: {}", meta.n_train_samples));
        lines.push(format!("  Model file:       {}", model_path.display()));

        let file_size = std::fs::metadata(&model_path)
            .map(|m| format_bytes(m.len()))
            .unwrap_or_else(|_| "unknown".into());
        lines.push(format!("  File size:        {file_size}"));

        self.show_results("Model Inspection", lines);
    }

    fn show_results(&mut self, title: &str, lines: Vec<String>) {
        self.result_title = title.to_string();
        self.result_lines = lines;
        self.result_is_error = false;
        self.result_scroll = 0;
        self.screen = Screen::Results;
    }

    fn show_error(&mut self, title: &str, message: &str) {
        self.result_title = title.to_string();
        self.result_lines = message.lines().map(|l| l.to_string()).collect();
        self.result_is_error = true;
        self.result_scroll = 0;
        self.screen = Screen::Results;
    }
}

fn run_app(terminal: &mut Terminal<CrosstermBackend<Stdout>>) -> anyhow::Result<()> {
    let mut app = App::new();

    loop {
        terminal.draw(|frame| ui::draw(frame, &app))?;

        let event = next_event()?;
        app.handle_event(event);

        if app.should_quit {
            break;
        }
    }

    Ok(())
}

pub fn algorithm_cli_name(algo: Algorithm) -> String {
    match algo {
        Algorithm::LogisticRegression => "logistic-regression".into(),
        Algorithm::KnnClassifier => "knn-classifier".into(),
        Algorithm::DecisionTreeClassifier => "decision-tree-classifier".into(),
        Algorithm::RandomForestClassifier => "random-forest-classifier".into(),
        Algorithm::GaussianNb => "gaussian-nb".into(),
        Algorithm::LinearRegression => "linear-regression".into(),
        Algorithm::RidgeRegression => "ridge-regression".into(),
        Algorithm::LassoRegression => "lasso-regression".into(),
        Algorithm::ElasticNet => "elastic-net".into(),
        Algorithm::KnnRegressor => "knn-regressor".into(),
        Algorithm::DecisionTreeRegressor => "decision-tree-regressor".into(),
        Algorithm::RandomForestRegressor => "random-forest-regressor".into(),
    }
}

pub fn algorithm_alias(algo: Algorithm) -> &'static str {
    match algo {
        Algorithm::LogisticRegression => "lr",
        Algorithm::KnnClassifier => "knn",
        Algorithm::DecisionTreeClassifier => "dtc",
        Algorithm::RandomForestClassifier => "rfc",
        Algorithm::GaussianNb => "gnb",
        Algorithm::LinearRegression => "linreg",
        Algorithm::RidgeRegression => "ridge",
        Algorithm::LassoRegression => "lasso",
        Algorithm::ElasticNet => "enet",
        Algorithm::KnnRegressor => "knnr",
        Algorithm::DecisionTreeRegressor => "dtr",
        Algorithm::RandomForestRegressor => "rfr",
    }
}

fn parse_algorithm(name: &str) -> Result<Algorithm, String> {
    let lower = name.to_lowercase();
    for algo in Algorithm::all() {
        if algorithm_cli_name(*algo) == lower || algorithm_alias(*algo) == lower {
            return Ok(*algo);
        }
    }
    Err(format!("Unknown algorithm: '{name}'"))
}

fn format_bytes(bytes: u64) -> String {
    const KIB: u64 = 1024;
    const MIB: u64 = KIB * 1024;
    const GIB: u64 = MIB * 1024;

    if bytes >= GIB {
        format!("{:.2} GiB", bytes as f64 / GIB as f64)
    } else if bytes >= MIB {
        format!("{:.2} MiB", bytes as f64 / MIB as f64)
    } else if bytes >= KIB {
        format!("{:.2} KiB", bytes as f64 / KIB as f64)
    } else {
        format!("{bytes} B")
    }
}

fn write_predictions_to_file(path: &PathBuf, predictions: &[f64]) -> Result<(), String> {
    use std::fs;
    use std::io::Write;

    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
    }

    let mut file = fs::File::create(path).map_err(|e| e.to_string())?;
    for pred in predictions {
        writeln!(file, "{pred}").map_err(|e| e.to_string())?;
    }
    Ok(())
}

fn load_all_columns(
    path: &PathBuf,
    expected_features: usize,
) -> Result<(smartcore::linalg::basic::matrix::DenseMatrix<f64>, usize), String> {
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_path(path)
        .map_err(|e| e.to_string())?;

    let headers: Vec<String> = reader
        .headers()
        .map_err(|e| e.to_string())?
        .iter()
        .map(|h| h.to_string())
        .collect();
    let n_features = headers.len();

    if n_features == 0 {
        return Err("CSV file has no columns".into());
    }

    let mut rows: Vec<Vec<f64>> = Vec::new();

    for (line_no, result) in reader.records().enumerate() {
        let record = result.map_err(|e| e.to_string())?;
        let mut row = Vec::with_capacity(n_features);
        for (col_idx, col_name) in headers.iter().enumerate() {
            let raw = record
                .get(col_idx)
                .ok_or_else(|| format!("row {line_no}: missing column index {col_idx}"))?;
            let val: f64 = raw.trim().parse().map_err(|_| {
                format!("row {line_no}, column '{col_name}': cannot parse '{raw}' as a number")
            })?;
            row.push(val);
        }
        rows.push(row);
    }

    if rows.is_empty() {
        return Err("CSV file contains no data rows".into());
    }

    if n_features != expected_features {
        return Err(format!(
            "model expects {expected_features} features but prediction CSV has {n_features} columns"
        ));
    }

    let n_rows = rows.len();

    let mut col_major = vec![0.0; n_rows * n_features];
    for (r, row) in rows.iter().enumerate() {
        for (c, &val) in row.iter().enumerate() {
            col_major[c * n_rows + r] = val;
        }
    }

    use smartcore::linalg::basic::matrix::DenseMatrix;
    let x = DenseMatrix::new(n_rows, n_features, col_major, true)
        .map_err(|e| format!("failed to build DenseMatrix: {e}"))?;

    Ok((x, n_features))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_algorithm_roundtrips() {
        assert_eq!(
            parse_algorithm("logistic-regression").unwrap(),
            Algorithm::LogisticRegression
        );
        assert_eq!(parse_algorithm("knn").unwrap(), Algorithm::KnnClassifier);
        assert_eq!(
            parse_algorithm("KNN-Classifier").unwrap(),
            Algorithm::KnnClassifier
        );
        assert!(parse_algorithm("magic-tree").is_err());

        // every algorithm should roundtrip through name and alias
        for algo in Algorithm::all() {
            let name = algorithm_cli_name(*algo);
            let alias = algorithm_alias(*algo);
            assert!(!name.is_empty());
            assert!(!alias.is_empty());
            assert_eq!(parse_algorithm(&name).unwrap(), *algo);
            assert_eq!(parse_algorithm(alias).unwrap(), *algo);
        }
    }

    #[test]
    fn format_bytes_units() {
        assert_eq!(format_bytes(0), "0 B");
        assert_eq!(format_bytes(512), "512 B");
        assert!(format_bytes(1536).contains("KiB"));
        assert!(format_bytes(2 * 1024 * 1024).contains("MiB"));
        assert!(format_bytes(3 * 1024 * 1024 * 1024).contains("GiB"));
    }

    #[test]
    fn text_field_editing() {
        let mut field = FormField::text("test", "hint", false);
        field.insert_char('a');
        field.insert_char('b');
        field.insert_char('c');
        assert_eq!(field.value, "abc");
        assert_eq!(field.cursor, 3);

        field.delete_char_before();
        assert_eq!(field.value, "ab");

        field.cursor_left();
        field.delete_char_at();
        assert_eq!(field.value, "a");
    }

    #[test]
    fn select_field_navigation() {
        let mut field = FormField::select(
            "algo",
            vec!["a".into(), "b".into(), "c".into()],
            vec!["desc a".into(), "desc b".into(), "desc c".into()],
        );
        assert_eq!(field.effective_value(), "a");

        field.cursor_right();
        assert_eq!(field.effective_value(), "b");
        field.cursor_right();
        assert_eq!(field.effective_value(), "c");
        field.cursor_right(); // shouldn't go past end
        assert_eq!(field.choice_idx, 2);

        field.cursor_home();
        assert_eq!(field.choice_idx, 0);
        field.cursor_end();
        assert_eq!(field.choice_idx, 2);
    }

    #[test]
    fn required_field_check() {
        let empty = FormField::text("label", "hint", true);
        assert!(!empty.is_filled());

        let mut filled = FormField::text("label", "hint", true);
        filled.insert_char('x');
        assert!(filled.is_filled());
    }

    #[test]
    fn menu_navigation_and_quit() {
        let mut app = App::new();
        assert_eq!(app.menu_index, 0);

        app.handle_event(AppEvent::Down);
        assert_eq!(app.menu_index, 1);
        app.handle_event(AppEvent::Up);
        assert_eq!(app.menu_index, 0);

        // wraps around
        app.handle_event(AppEvent::Up);
        assert_eq!(app.menu_index, MENU_ITEMS.len() - 1);
        app.handle_event(AppEvent::Down);
        assert_eq!(app.menu_index, 0);

        // q quits
        app.handle_event(AppEvent::Char('q'));
        assert!(app.should_quit);
    }

    #[test]
    fn menu_enter_and_escape() {
        let mut app = App::new();
        app.menu_index = 4;
        app.handle_event(AppEvent::Enter);
        assert_eq!(app.screen, Screen::AlgorithmsList);

        app.handle_event(AppEvent::Escape);
        assert_eq!(app.screen, Screen::MainMenu);
    }

    #[test]
    fn ctrl_c_always_quits() {
        let mut app = App::new();
        app.goto(Screen::AlgorithmsList);
        app.handle_event(AppEvent::CtrlC);
        assert!(app.should_quit);
    }

    #[test]
    fn form_focus_cycling_and_escape() {
        let mut app = App::new();
        app.goto(Screen::TrainForm);
        assert_eq!(app.form_focus, 0);

        // tab through all fields to the submit button
        let n = app.form_fields.len();
        for _ in 0..n {
            app.handle_event(AppEvent::Tab);
        }
        assert!(app.submit_focused);

        // one more wraps back
        app.handle_event(AppEvent::Tab);
        assert_eq!(app.form_focus, 0);
        assert!(!app.submit_focused);

        // escape goes back to menu
        app.handle_event(AppEvent::Escape);
        assert_eq!(app.screen, Screen::MainMenu);
    }

    #[test]
    fn number_shortcuts() {
        let mut app = App::new();
        app.handle_event(AppEvent::Char('1'));
        assert_eq!(app.screen, Screen::TrainForm);

        app.back_to_menu();
        app.handle_event(AppEvent::Char('5'));
        assert_eq!(app.screen, Screen::AlgorithmsList);
    }
}
