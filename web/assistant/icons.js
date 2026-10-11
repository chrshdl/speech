// Weather icons as SVG markup, drawn on a 64 by 64 grid.

// Colors that show on the cards' white and pastel backgrounds.
const ICON_COLORS = {
  sun: "#ffbf2e",
  moon: "#b9a3dc",
  cloud: "#bcd3f2",
  darkCloud: "#8d98b4",
  rain: "#4aa3ff",
  snow: "#7fc4ff",
  bolt: "#ffbf2e",
  fog: "#9fb0cc",
};

function cloud(fill = ICON_COLORS.cloud, transform = "") {
  return `<g fill="${fill}" transform="${transform}">
    <circle cx="22" cy="38" r="10"/><circle cx="34" cy="30" r="13"/>
    <circle cx="45" cy="38" r="10"/><rect x="22" y="36" width="23" height="12"/>
  </g>`;
}

function sun(cx = 32, cy = 32, r = 11) {
  const rays = [];
  for (let i = 0; i < 8; i++) {
    const angle = (i * Math.PI) / 4;
    const [x1, y1] = [cx + Math.cos(angle) * (r + 5), cy + Math.sin(angle) * (r + 5)];
    const [x2, y2] = [cx + Math.cos(angle) * (r + 10), cy + Math.sin(angle) * (r + 10)];
    rays.push(`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}"/>`);
  }
  return `<g stroke="${ICON_COLORS.sun}" stroke-width="3.5" stroke-linecap="round">${rays.join("")}</g>
    <circle cx="${cx}" cy="${cy}" r="${r}" fill="${ICON_COLORS.sun}"/>`;
}

function moon(cx = 30, cy = 32, r = 16) {
  // A crescent: the disc without a smaller disc to its upper right.
  const id = `moon-${cx}-${cy}-${r}`;
  return `<mask id="${id}"><rect width="64" height="64" fill="#fff"/>
      <circle cx="${cx + r * 0.55}" cy="${cy - r * 0.45}" r="${r * 0.82}" fill="#000"/></mask>
    <circle cx="${cx}" cy="${cy}" r="${r}" fill="${ICON_COLORS.moon}" mask="url(#${id})"/>`;
}

function lines(color, segments, width = 3.5) {
  const drawn = segments.map(([x1, y1, x2, y2]) => `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}"/>`);
  return `<g stroke="${color}" stroke-width="${width}" stroke-linecap="round">${drawn.join("")}</g>`;
}

function dots(color, points, r) {
  return points.map(([x, y]) => `<circle cx="${x}" cy="${y}" r="${r}" fill="${color}"/>`).join("");
}

const RAISED = "translate(0 -7)";

const ICONS = {
  CLEAR: (day) => (day ? sun() : moon()),
  PARTLY_CLOUDY: (day) => (day ? sun(23, 22, 8) : moon(23, 22, 11)) + cloud(ICON_COLORS.cloud, "translate(7 6) scale(0.86)"),
  CLOUDY: () => cloud(ICON_COLORS.darkCloud, "translate(14 4) scale(0.7)") + cloud(ICON_COLORS.cloud, "translate(-2 4)"),
  FOG: () =>
    cloud(ICON_COLORS.cloud, RAISED) +
    lines(ICON_COLORS.fog, [
      [14, 49, 44, 49],
      [22, 56, 52, 56],
    ]),
  DRIZZLE: () => cloud(ICON_COLORS.cloud, RAISED) + dots(ICON_COLORS.rain, [[24, 50], [34, 54], [44, 50]], 2.5),
  RAIN: () =>
    cloud(ICON_COLORS.cloud, RAISED) +
    lines(ICON_COLORS.rain, [
      [25, 47, 22, 55],
      [35, 47, 32, 55],
      [45, 47, 42, 55],
    ]),
  SNOW: () =>
    cloud(ICON_COLORS.cloud, RAISED) +
    dots(ICON_COLORS.snow, [[23, 49], [33, 49], [43, 49], [28, 56], [38, 56]], 2.6),
  THUNDERSTORM: () =>
    cloud(ICON_COLORS.darkCloud, RAISED) +
    `<polygon points="35,38 26,51 32,51 28,61 41,46 35,46 39,38" fill="${ICON_COLORS.bolt}"/>`,
};

// The SVG of the condition, such as "RAIN", at day or at night.
function weatherIcon(condition, isDay = true) {
  const draw = ICONS[condition] || ICONS.CLOUDY;
  return `<svg class="weather-icon" viewBox="0 0 64 64" aria-hidden="true">${draw(isDay)}</svg>`;
}
