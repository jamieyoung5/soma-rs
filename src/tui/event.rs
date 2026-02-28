use std::time::Duration;

use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyModifiers};

const TICK_RATE: Duration = Duration::from_millis(50);

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AppEvent {
    Char(char),
    Backspace,
    Delete,
    Enter,
    Escape,
    Tab,
    BackTab,
    Up,
    Down,
    Left,
    Right,
    Home,
    End,
    CtrlC,
    F1,
    Tick,
}

/// Poll for the next event, returning `Tick` if nothing happens within the
/// polling window.
pub fn next_event() -> std::io::Result<AppEvent> {
    if event::poll(TICK_RATE)? {
        if let Event::Key(key) = event::read()? {
            return Ok(translate_key(key));
        }
    }
    Ok(AppEvent::Tick)
}

fn translate_key(key: KeyEvent) -> AppEvent {
    if key.modifiers.contains(KeyModifiers::CONTROL) {
        return match key.code {
            KeyCode::Char('c') => AppEvent::CtrlC,
            KeyCode::Char(c) => AppEvent::Char(c),
            _ => AppEvent::Tick,
        };
    }

    match key.code {
        KeyCode::Char(c) => AppEvent::Char(c),
        KeyCode::Backspace => AppEvent::Backspace,
        KeyCode::Delete => AppEvent::Delete,
        KeyCode::Enter => AppEvent::Enter,
        KeyCode::Esc => AppEvent::Escape,
        KeyCode::Tab => AppEvent::Tab,
        KeyCode::BackTab => AppEvent::BackTab,
        KeyCode::Up => AppEvent::Up,
        KeyCode::Down => AppEvent::Down,
        KeyCode::Left => AppEvent::Left,
        KeyCode::Right => AppEvent::Right,
        KeyCode::Home => AppEvent::Home,
        KeyCode::End => AppEvent::End,
        KeyCode::F(1) => AppEvent::F1,
        _ => AppEvent::Tick,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossterm::event::{KeyEventKind, KeyEventState};

    fn make_key(code: KeyCode, modifiers: KeyModifiers) -> KeyEvent {
        KeyEvent {
            code,
            modifiers,
            kind: KeyEventKind::Press,
            state: KeyEventState::NONE,
        }
    }

    #[test]
    fn key_translation() {
        assert_eq!(
            translate_key(make_key(KeyCode::Char('c'), KeyModifiers::CONTROL)),
            AppEvent::CtrlC
        );
        assert_eq!(
            translate_key(make_key(KeyCode::Char('a'), KeyModifiers::NONE)),
            AppEvent::Char('a')
        );
        assert_eq!(
            translate_key(make_key(KeyCode::Enter, KeyModifiers::NONE)),
            AppEvent::Enter
        );
        assert_eq!(
            translate_key(make_key(KeyCode::Esc, KeyModifiers::NONE)),
            AppEvent::Escape
        );
        assert_eq!(
            translate_key(make_key(KeyCode::F(1), KeyModifiers::NONE)),
            AppEvent::F1
        );
        // Unknown keys map to Tick
        assert_eq!(
            translate_key(make_key(KeyCode::F(12), KeyModifiers::NONE)),
            AppEvent::Tick
        );
    }

    #[test]
    fn arrows_and_nav() {
        assert_eq!(
            translate_key(make_key(KeyCode::Up, KeyModifiers::NONE)),
            AppEvent::Up
        );
        assert_eq!(
            translate_key(make_key(KeyCode::Down, KeyModifiers::NONE)),
            AppEvent::Down
        );
        assert_eq!(
            translate_key(make_key(KeyCode::Left, KeyModifiers::NONE)),
            AppEvent::Left
        );
        assert_eq!(
            translate_key(make_key(KeyCode::Right, KeyModifiers::NONE)),
            AppEvent::Right
        );
        assert_eq!(
            translate_key(make_key(KeyCode::Home, KeyModifiers::NONE)),
            AppEvent::Home
        );
        assert_eq!(
            translate_key(make_key(KeyCode::End, KeyModifiers::NONE)),
            AppEvent::End
        );
    }

    #[test]
    fn tab_variants() {
        assert_eq!(
            translate_key(make_key(KeyCode::Tab, KeyModifiers::NONE)),
            AppEvent::Tab
        );
        assert_eq!(
            translate_key(make_key(KeyCode::BackTab, KeyModifiers::SHIFT)),
            AppEvent::BackTab
        );
    }

    #[test]
    fn backspace() {
        assert_eq!(
            translate_key(make_key(KeyCode::Backspace, KeyModifiers::NONE)),
            AppEvent::Backspace
        );
    }
}
