// Welcome lines for the start screen (English by design). One is drawn at
// random on page load and on every new conversation.

export const GREETINGS = [
  "Good day",
  "Hello there",
  "Welcome back",
  "What's on your mind?",
  "Let's get into it",
  "Let's dig in",
  "Ready when you are",
  "Where shall we start?",
  "Curious about something?",
  "Ask away",
  "Got a question?",
  "What are we exploring today?",
  "Let's figure it out",
  "Let's think it through",
  "Let's make sense of it",
  "Pick a thread",
  "Fresh page, fresh ideas",
  "What would you like to know?",
  "Shall we begin?",
  "Wonder out loud",
];

export function timeGreeting(hour) {
  if (hour >= 5 && hour < 12) return "Good morning";
  if (hour >= 12 && hour < 18) return "Good afternoon";
  return "Good evening";
}

/** Random greeting, never the same as `previous` twice in a row. */
export function pickGreeting(previous = "", { random = Math.random, hour = new Date().getHours() } = {}) {
  const pool = [...GREETINGS, timeGreeting(hour)].filter((g) => g !== previous);
  return pool[Math.floor(random() * pool.length) % pool.length];
}
