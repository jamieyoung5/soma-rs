use ratatui::layout::{Alignment, Constraint, Direction, Layout, Margin, Rect};
use ratatui::style::{Color, Modifier, Style, Stylize};
use ratatui::text::{Line, Span};
use ratatui::widgets::{
    Block, BorderType, Borders, List, ListItem, Padding, Paragraph, Scrollbar,
    ScrollbarOrientation, ScrollbarState, Wrap,
};
use ratatui::Frame;
use tui_big_text::{BigText, PixelSize};

use crate::models::Algorithm;

use super::app::{algorithm_alias, algorithm_cli_name, App, FormField, Screen, MENU_ITEMS};

const BRAND: Color = Color::Rgb(130, 170, 255);
const ACCENT: Color = Color::Rgb(180, 230, 140);
const WARN: Color = Color::Rgb(255, 180, 100);
const ERR: Color = Color::Rgb(255, 110, 110);
const DIM: Color = Color::DarkGray;
const SURFACE: Color = Color::Rgb(40, 42, 54);
const FIELD_BG: Color = Color::Rgb(50, 52, 64);
const SELECTED_BG: Color = Color::Rgb(60, 70, 100);
const TEXT: Color = Color::Rgb(220, 220, 220);
const MUTED: Color = Color::Rgb(140, 140, 160);

pub fn draw(frame: &mut Frame, app: &App) {
    let size = frame.area();
    let bg = Block::default().style(Style::default().bg(SURFACE));
    frame.render_widget(bg, size);

    match app.screen {
        Screen::MainMenu => draw_main_menu(frame, app, size),
        Screen::TrainForm => draw_form(frame, app, size, "Train a Model"),
        Screen::PredictForm => draw_form(frame, app, size, "Make Predictions"),
        Screen::EvaluateForm => draw_form(frame, app, size, "Evaluate a Model"),
        Screen::InspectForm => draw_form(frame, app, size, "Inspect a Model"),
        Screen::AlgorithmsList => draw_algorithms(frame, app, size),
        Screen::Results => draw_results(frame, app, size),
        Screen::Help => draw_help(frame, app, size),
    }
}

fn draw_main_menu(frame: &mut Frame, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(7), // logo/header
            Constraint::Min(10),   // menu
            Constraint::Length(3), // status bar
        ])
        .split(area);

    draw_header(frame, chunks[0]);

    let menu_block = Block::default()
        .title(Line::from(" Actions ").fg(BRAND).bold())
        .title_alignment(Alignment::Left)
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(DIM))
        .padding(Padding::new(2, 2, 1, 1));

    let inner = menu_block.inner(chunks[1]);
    frame.render_widget(menu_block, chunks[1]);

    // Center the menu list within the inner area.
    let menu_area = centered_rect_fixed(46, MENU_ITEMS.len() as u16 + 2, inner);

    let items: Vec<ListItem> = MENU_ITEMS
        .iter()
        .enumerate()
        .map(|(i, label)| {
            let number = if i < 6 {
                format!(" {}  ", i + 1)
            } else {
                "    ".to_string()
            };

            let is_selected = i == app.menu_index;

            let line = if is_selected {
                Line::from(vec![
                    Span::styled(
                        " ▸ ",
                        Style::default().fg(BRAND).add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(number, Style::default().fg(MUTED)),
                    Span::styled(
                        *label,
                        Style::default()
                            .fg(Color::White)
                            .add_modifier(Modifier::BOLD),
                    ),
                ])
            } else {
                Line::from(vec![
                    Span::styled("   ", Style::default()),
                    Span::styled(number, Style::default().fg(DIM)),
                    Span::styled(*label, Style::default().fg(TEXT)),
                ])
            };

            let style = if is_selected {
                Style::default().bg(SELECTED_BG)
            } else {
                Style::default()
            };

            ListItem::new(line).style(style)
        })
        .collect();

    let list = List::new(items);
    frame.render_widget(list, menu_area);

    draw_status_bar(frame, &app.status_message, chunks[2]);
}

fn draw_header(frame: &mut Frame, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(4), // big text
            Constraint::Length(1), // subtitle
            Constraint::Min(0),    // spacing
        ])
        .split(area);

    let big_title = BigText::builder()
        .pixel_size(PixelSize::HalfHeight)
        .style(Style::default().fg(BRAND).bold())
        .lines(vec![Line::from("SOMA")])
        .alignment(Alignment::Center)
        .build();
    frame.render_widget(big_title, chunks[0]);

    let subtitle = Paragraph::new(Line::from(Span::styled(
        "ML Toolkit  ·  powered by smartcore",
        Style::default().fg(DIM),
    )))
    .alignment(Alignment::Center);
    frame.render_widget(subtitle, chunks[1]);
}

fn draw_form(frame: &mut Frame, app: &App, area: Rect, title: &str) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // title
            Constraint::Min(6),    // form fields
            Constraint::Length(3), // status bar
        ])
        .split(area);

    let title_para = Paragraph::new(Line::from(vec![
        Span::styled(" ◆ ", Style::default().fg(ACCENT)),
        Span::styled(title, Style::default().fg(Color::White).bold()),
    ]))
    .alignment(Alignment::Center)
    .block(
        Block::default()
            .borders(Borders::BOTTOM)
            .border_style(Style::default().fg(DIM)),
    );
    frame.render_widget(title_para, chunks[0]);

    let form_block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(DIM))
        .padding(Padding::new(2, 2, 1, 1));

    let form_inner = form_block.inner(chunks[1]);
    frame.render_widget(form_block, chunks[1]);

    // Each field takes 3 rows: label, input, gap.
    let n_fields = app.form_fields.len();
    let total_rows = n_fields * 3 + 2; // +2 for submit button area

    let field_area = centered_rect_fixed(
        form_inner.width.min(70),
        (total_rows as u16).min(form_inner.height),
        form_inner,
    );

    let mut constraints: Vec<Constraint> = Vec::with_capacity(n_fields + 1);
    for _ in 0..n_fields {
        constraints.push(Constraint::Length(3));
    }
    constraints.push(Constraint::Length(2)); // submit button

    let field_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints(constraints)
        .split(field_area);

    // Render each field.
    for (i, field) in app.form_fields.iter().enumerate() {
        if i >= field_chunks.len() - 1 {
            break;
        }
        let is_focused = !app.submit_focused && app.form_focus == i;
        draw_form_field(frame, field, field_chunks[i], is_focused);
    }

    let submit_idx = field_chunks.len() - 1;
    if submit_idx < field_chunks.len() {
        let submit_style = if app.submit_focused {
            Style::default()
                .fg(Color::Black)
                .bg(ACCENT)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(ACCENT)
        };

        let submit_text = if app.submit_focused {
            " ▸ Submit (Enter) "
        } else {
            "   Submit (Enter)  "
        };

        let submit = Paragraph::new(submit_text)
            .style(submit_style)
            .alignment(Alignment::Center);
        frame.render_widget(submit, field_chunks[submit_idx]);
    }

    let hint = "Tab/↓: next field • Shift+Tab/↑: previous • Esc: back to menu";
    draw_status_bar(frame, hint, chunks[2]);
}

fn draw_form_field(frame: &mut Frame, field: &FormField, area: Rect, focused: bool) {
    if area.height < 2 {
        return;
    }

    let label_area = Rect {
        x: area.x,
        y: area.y,
        width: area.width,
        height: 1,
    };
    let input_area = Rect {
        x: area.x,
        y: area.y + 1,
        width: area.width,
        height: 1.min(area.height.saturating_sub(1)),
    };

    // Label.
    let required_marker = if field.required { "*" } else { "" };
    let label_style = if focused {
        Style::default().fg(BRAND).add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(MUTED)
    };
    let label = Paragraph::new(Line::from(vec![
        Span::styled(&field.label, label_style),
        Span::styled(required_marker, Style::default().fg(WARN)),
    ]));
    frame.render_widget(label, label_area);

    // Input.
    if field.choices.is_some() {
        // Selection field: show current choice with ◂ ▸ arrows.
        let border_color = if focused { BRAND } else { DIM };
        let selected = field.effective_value();
        let count = field.choice_count();
        let pos = field.choice_idx + 1;
        let desc = field.selected_description().unwrap_or("");

        let prefix = if focused {
            Span::styled("│ ", Style::default().fg(border_color))
        } else {
            Span::styled("  ", Style::default())
        };

        let left_arrow = if focused && field.choice_idx > 0 {
            Span::styled("◂  ", Style::default().fg(BRAND))
        } else {
            Span::styled("   ", Style::default().fg(DIM))
        };

        let name_style = if focused {
            Style::default()
                .fg(Color::White)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(TEXT)
        };

        let right_arrow = if focused && field.choice_idx + 1 < count {
            Span::styled("  ▸", Style::default().fg(BRAND))
        } else {
            Span::styled("   ", Style::default().fg(DIM))
        };

        let position = Span::styled(format!("   {pos}/{count}"), Style::default().fg(DIM));

        let desc_span = if !desc.is_empty() {
            Span::styled(format!("   {desc}"), Style::default().fg(MUTED).italic())
        } else {
            Span::raw("")
        };

        let line = Line::from(vec![
            prefix,
            left_arrow,
            Span::styled(selected, name_style),
            right_arrow,
            position,
            desc_span,
        ]);

        let input_para = Paragraph::new(line).style(Style::default().bg(FIELD_BG));
        frame.render_widget(input_para, input_area);
    } else {
        // Text input field.
        let border_color = if focused { BRAND } else { DIM };
        let display_text = if field.value.is_empty() {
            Span::styled(&field.placeholder, Style::default().fg(DIM).italic())
        } else {
            Span::styled(&field.value, Style::default().fg(TEXT))
        };

        let prefix = if focused {
            Span::styled("│ ", Style::default().fg(border_color))
        } else {
            Span::styled("  ", Style::default())
        };

        let input_line = Line::from(vec![prefix, display_text]);
        let input_para = Paragraph::new(input_line).style(Style::default().bg(FIELD_BG));
        frame.render_widget(input_para, input_area);

        // Show cursor for focused text fields.
        if focused {
            let cursor_x = input_area.x + 2 + field.cursor as u16;
            let cursor_y = input_area.y;
            if cursor_x < input_area.x + input_area.width {
                frame.set_cursor_position((cursor_x, cursor_y));
            }
        }
    }
}

fn draw_algorithms(frame: &mut Frame, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // title
            Constraint::Min(8),    // list
            Constraint::Length(3), // status
        ])
        .split(area);

    let title = Paragraph::new(Line::from(vec![
        Span::styled(" ◆ ", Style::default().fg(ACCENT)),
        Span::styled(
            "Supported Algorithms",
            Style::default().fg(Color::White).bold(),
        ),
    ]))
    .alignment(Alignment::Center)
    .block(
        Block::default()
            .borders(Borders::BOTTOM)
            .border_style(Style::default().fg(DIM)),
    );
    frame.render_widget(title, chunks[0]);

    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(DIM))
        .padding(Padding::new(1, 1, 0, 0));
    let inner = block.inner(chunks[1]);
    frame.render_widget(block, chunks[1]);

    let all_algos = Algorithm::all();

    let header = Line::from(vec![
        Span::styled(
            format!(" {:<28}", "Name"),
            Style::default().fg(BRAND).add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            format!("{:<8}", "Alias"),
            Style::default().fg(BRAND).add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            format!("{:<16}", "Task"),
            Style::default().fg(BRAND).add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            "Description",
            Style::default().fg(BRAND).add_modifier(Modifier::BOLD),
        ),
    ]);

    let separator = Line::from(Span::styled(
        " ─".to_string() + &"─".repeat(inner.width.saturating_sub(3) as usize),
        Style::default().fg(DIM),
    ));

    let mut text_lines: Vec<Line> = Vec::with_capacity(all_algos.len() + 3);
    text_lines.push(header);
    text_lines.push(separator);

    for (i, algo) in all_algos.iter().enumerate() {
        let name = algorithm_cli_name(*algo);
        let alias = algorithm_alias(*algo);
        let task = algo.task().to_string();
        let desc = algo.description();

        let is_highlighted = i == app.algo_scroll;

        let style = if is_highlighted {
            Style::default().bg(SELECTED_BG)
        } else {
            Style::default()
        };

        let task_color = if task == "classification" {
            BRAND
        } else {
            ACCENT
        };

        let indicator = if is_highlighted { " ▸ " } else { "   " };

        let line = Line::from(vec![
            Span::styled(
                indicator,
                Style::default().fg(if is_highlighted { BRAND } else { DIM }),
            ),
            Span::styled(
                format!("{name:<25}"),
                Style::default().fg(TEXT).patch(style),
            ),
            Span::styled(
                format!("{alias:<8}"),
                Style::default().fg(WARN).patch(style),
            ),
            Span::styled(
                format!("{task:<16}"),
                Style::default().fg(task_color).patch(style),
            ),
            Span::styled(desc, Style::default().fg(MUTED).patch(style)),
        ]);

        text_lines.push(line);
    }

    let visible_height = inner.height.saturating_sub(2);
    let content_offset = if all_algos.len() as u16 > visible_height {
        let max_scroll = all_algos.len().saturating_sub(visible_height as usize);
        app.algo_scroll.min(max_scroll) as u16
    } else {
        0
    };

    let para = Paragraph::new(text_lines).scroll((content_offset, 0));
    frame.render_widget(para, inner);

    if all_algos.len() as u16 > visible_height {
        let scrollbar = Scrollbar::new(ScrollbarOrientation::VerticalRight);
        let mut scrollbar_state = ScrollbarState::new(all_algos.len()).position(app.algo_scroll);
        let scrollbar_area = chunks[1].inner(Margin {
            vertical: 1,
            horizontal: 0,
        });
        frame.render_stateful_widget(scrollbar, scrollbar_area, &mut scrollbar_state);
    }

    draw_status_bar(frame, "↑↓: scroll • Esc/q: back to menu", chunks[2]);
}

fn draw_results(frame: &mut Frame, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // title
            Constraint::Min(6),    // content
            Constraint::Length(3), // status
        ])
        .split(area);

    let title_color = if app.result_is_error { ERR } else { ACCENT };
    let title_icon = if app.result_is_error { "✗" } else { "✓" };

    let title = Paragraph::new(Line::from(vec![
        Span::styled(format!(" {title_icon} "), Style::default().fg(title_color)),
        Span::styled(
            &app.result_title,
            Style::default()
                .fg(Color::White)
                .add_modifier(Modifier::BOLD),
        ),
    ]))
    .alignment(Alignment::Center)
    .block(
        Block::default()
            .borders(Borders::BOTTOM)
            .border_style(Style::default().fg(DIM)),
    );
    frame.render_widget(title, chunks[0]);

    let content_block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(if app.result_is_error { ERR } else { DIM }))
        .padding(Padding::new(2, 2, 1, 1));

    let content_inner = content_block.inner(chunks[1]);
    frame.render_widget(content_block, chunks[1]);

    let lines: Vec<Line> = app
        .result_lines
        .iter()
        .map(|l| {
            if l.starts_with('✓') {
                Line::from(Span::styled(l.as_str(), Style::default().fg(ACCENT)))
            } else if l.contains("──") {
                Line::from(Span::styled(
                    l.as_str(),
                    Style::default().fg(BRAND).add_modifier(Modifier::BOLD),
                ))
            } else if l.starts_with("Warning:") {
                Line::from(Span::styled(l.as_str(), Style::default().fg(WARN)))
            } else if app.result_is_error {
                Line::from(Span::styled(l.as_str(), Style::default().fg(ERR)))
            } else {
                Line::from(Span::styled(l.as_str(), Style::default().fg(TEXT)))
            }
        })
        .collect();

    let max_scroll = lines.len().saturating_sub(content_inner.height as usize);
    let scroll = (app.result_scroll as usize).min(max_scroll) as u16;

    let para = Paragraph::new(lines.clone())
        .scroll((scroll, 0))
        .wrap(Wrap { trim: false });
    frame.render_widget(para, content_inner);

    if lines.len() as u16 > content_inner.height {
        let scrollbar = Scrollbar::new(ScrollbarOrientation::VerticalRight);
        let mut scrollbar_state = ScrollbarState::new(lines.len()).position(scroll as usize);
        let scrollbar_area = chunks[1].inner(Margin {
            vertical: 1,
            horizontal: 0,
        });
        frame.render_stateful_widget(scrollbar, scrollbar_area, &mut scrollbar_state);
    }

    let hint = if app.result_is_error {
        "Esc/Enter/q: back to menu"
    } else {
        "↑↓: scroll • Esc/Enter/q: back to menu"
    };
    draw_status_bar(frame, hint, chunks[2]);
}

fn draw_help(frame: &mut Frame, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // title
            Constraint::Min(10),   // content
            Constraint::Length(3), // status
        ])
        .split(area);

    let title = Paragraph::new(Line::from(vec![
        Span::styled(" ◆ ", Style::default().fg(ACCENT)),
        Span::styled(
            "Help & Keybindings",
            Style::default().fg(Color::White).bold(),
        ),
    ]))
    .alignment(Alignment::Center)
    .block(
        Block::default()
            .borders(Borders::BOTTOM)
            .border_style(Style::default().fg(DIM)),
    );
    frame.render_widget(title, chunks[0]);

    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(DIM))
        .padding(Padding::new(3, 3, 1, 1));
    let inner = block.inner(chunks[1]);
    frame.render_widget(block, chunks[1]);

    let help_lines = vec![
        Line::from(Span::styled(
            "Global Keybindings",
            Style::default().fg(BRAND).bold(),
        )),
        Line::from(""),
        help_row("Ctrl+C", "Quit immediately from any screen"),
        help_row("q / Q", "Quit / go back (context-dependent)"),
        help_row("Esc", "Go back to the main menu"),
        Line::from(""),
        Line::from(Span::styled("Main Menu", Style::default().fg(BRAND).bold())),
        Line::from(""),
        help_row("↑ / ↓", "Navigate menu items"),
        help_row("Enter", "Select the highlighted item"),
        help_row("1–6", "Jump directly to a menu item"),
        help_row("F1", "Open this help screen"),
        Line::from(""),
        Line::from(Span::styled(
            "Form Screens (Train / Predict / Evaluate / Inspect)",
            Style::default().fg(BRAND).bold(),
        )),
        Line::from(""),
        help_row("Tab / ↓", "Move to next field"),
        help_row("Shift+Tab / ↑", "Move to previous field"),
        help_row("← / →", "Move cursor in text fields"),
        help_row("← / →", "Cycle choices in selection fields"),
        help_row("Home / End", "Jump to start / end of field"),
        help_row("Enter", "Submit when the Submit button is focused"),
        help_row("Esc", "Cancel and return to menu"),
        Line::from(""),
        Line::from(Span::styled(
            "Results & List Screens",
            Style::default().fg(BRAND).bold(),
        )),
        Line::from(""),
        help_row("↑ / ↓", "Scroll content"),
        help_row("Home / End", "Jump to top / bottom"),
        help_row("Enter / Esc / q", "Return to menu"),
        Line::from(""),
        Line::from(Span::styled(
            "About soma",
            Style::default().fg(BRAND).bold(),
        )),
        Line::from(""),
        Line::from(Span::styled(
            "soma is a command-line ML toolkit powered by smartcore.",
            Style::default().fg(TEXT),
        )),
        Line::from(Span::styled(
            "It supports 12 algorithms for classification and regression tasks.",
            Style::default().fg(TEXT),
        )),
        Line::from(Span::styled(
            "All data must be in CSV format with numeric values and a header row.",
            Style::default().fg(TEXT),
        )),
        Line::from(""),
        Line::from(Span::styled(
            "Tip: You can also use soma from the command line with `soma --help`.",
            Style::default().fg(MUTED).italic(),
        )),
    ];

    let scroll = app
        .help_scroll
        .min(help_lines.len().saturating_sub(inner.height as usize) as u16);

    let para = Paragraph::new(help_lines.clone())
        .scroll((scroll, 0))
        .wrap(Wrap { trim: false });
    frame.render_widget(para, inner);

    if help_lines.len() as u16 > inner.height {
        let scrollbar = Scrollbar::new(ScrollbarOrientation::VerticalRight);
        let mut scrollbar_state = ScrollbarState::new(help_lines.len()).position(scroll as usize);
        let scrollbar_area = chunks[1].inner(Margin {
            vertical: 1,
            horizontal: 0,
        });
        frame.render_stateful_widget(scrollbar, scrollbar_area, &mut scrollbar_state);
    }

    draw_status_bar(frame, "↑↓: scroll • Esc/q/Enter: back to menu", chunks[2]);
}

fn help_row<'a>(key: &'a str, desc: &'a str) -> Line<'a> {
    Line::from(vec![
        Span::styled(format!("  {key:<20}"), Style::default().fg(WARN)),
        Span::styled(desc, Style::default().fg(TEXT)),
    ])
}

fn draw_status_bar(frame: &mut Frame, message: &str, area: Rect) {
    let status = Paragraph::new(Line::from(vec![
        Span::styled(" ◈ ", Style::default().fg(BRAND)),
        Span::styled(message, Style::default().fg(MUTED)),
    ]))
    .block(
        Block::default()
            .borders(Borders::TOP)
            .border_style(Style::default().fg(DIM)),
    );
    frame.render_widget(status, area);
}

fn centered_rect_fixed(width: u16, height: u16, area: Rect) -> Rect {
    let w = width.min(area.width);
    let h = height.min(area.height);
    let x = area.x + (area.width.saturating_sub(w)) / 2;
    let y = area.y + (area.height.saturating_sub(h)) / 2;
    Rect::new(x, y, w, h)
}
