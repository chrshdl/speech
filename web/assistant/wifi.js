// Joining a Wi-Fi network, where the service can, see /v1/wifi in
// docs/backend-api.md: pick a network, type its password on the keyboard
// below it, as a touchscreen without a keyboard of its own needs, and wait
// while the service joins it.

// How often the button checks whether a network is joined, in ms.
const WIFI_CHECK_MS = 10000;
// How long "you're online" shows before the sheet closes, in ms.
const WIFI_JOINED_MS = 2600;

const WIFI_ERRORS = {
  WRONG_PASSWORD: () => "That password didn’t work. Let’s try again!",
  NOT_JOINED: (ssid) => `${ssid} didn’t answer. Is it close enough?`,
  NO_ADDRESS: (ssid) => `Joined ${ssid}, but it gave no address. Try again in a bit.`,
};

const LOCK_ICON = `<svg class="wifi-lock" viewBox="0 0 24 24" aria-label="Needs a password">
  <rect x="5" y="10.5" width="14" height="10" rx="3" /><path d="M8.5 10.5V8a3.5 3.5 0 0 1 7 0v2.5" /></svg>`;

function signalIcon(bars) {
  const heights = [6, 10, 14, 18];
  return `<svg class="wifi-signal" viewBox="0 0 24 24" aria-label="Signal ${bars} of 4">
    ${heights
      .map((h, i) => `<rect x="${2.5 + i * 5.5}" y="${21 - h}" width="3.6" height="${h}" rx="1.8" class="${i < bars ? "on" : ""}" />`)
      .join("")}</svg>`;
}

class WifiSetup {
  // The elements: the sheet's section and the bar's button.
  constructor({ root, button }) {
    this.root = root;
    this.button = button;
    const $ = (selector) => root.querySelector(selector);
    this.ui = {
      title: $("#wifi-title-text"),
      back: $("#wifi-back"),
      close: $("#wifi-close"),
      networks: $("#wifi-networks"),
      note: $("#wifi-networks-note"),
      rescan: $("#wifi-rescan"),
      other: $("#wifi-other"),
      form: $("#wifi-form"),
      ssidField: $("#wifi-ssid-field"),
      ssid: $("#wifi-ssid"),
      passwordField: $("#wifi-password-field"),
      password: $("#wifi-password"),
      reveal: $("#wifi-reveal"),
      error: $("#wifi-error"),
      joiningText: $("#wifi-joining-text"),
      joinedText: $("#wifi-joined-text"),
    };
    this.status = null; // The last WifiStatus.
    this.networks = []; // The networks of the last scan.
    this.network = null; // The network picked: its ssid and whether it is secured.
    this.focused = null; // The input the keyboard types into.
    this.scanning = false;
    // Counts the joins, so one the user walked away from cannot take over
    // the sheet when it finishes.
    this.attempt = 0;
    this.closeTimer = null;
    this.keyboard = new OnScreenKeyboard($("#keyboard"), {
      type: (text) => this.type(text),
      erase: (all) => this.erase(all),
      enter: () => this.submit(),
    });

    button.addEventListener("click", () => this.open());
    this.ui.close.addEventListener("click", () => this.close());
    this.ui.back.addEventListener("click", () => this.back());
    this.ui.rescan.addEventListener("click", () => this.scan());
    this.ui.other.addEventListener("click", () => this.showPassword(null));
    this.ui.reveal.addEventListener("click", () => this.toggleReveal());
    this.ui.form.addEventListener("submit", (event) => {
      event.preventDefault();
      this.submit();
    });
    for (const input of [this.ui.ssid, this.ui.password]) {
      input.addEventListener("focus", () => this.focus(input));
      input.addEventListener("input", () => this.showError(""));
    }
    // A tap on the sheet's backdrop closes it.
    root.addEventListener("click", (event) => {
      if (event.target === root) {
        this.close();
      }
    });
  }

  get isOpen() {
    return !this.root.hidden;
  }

  // Shows the button, and the sheet when no network is saved to join.
  async start() {
    this.button.hidden = false;
    await this.check();
    setInterval(() => this.check(), WIFI_CHECK_MS);
    if (this.status && this.status.state !== "CONNECTED" && !this.status.configured) {
      this.open();
    }
  }

  async check() {
    try {
      const response = await fetch("/v1/wifi");
      this.status = response.ok ? await response.json() : null;
    } catch {
      this.status = null;
    }
    const state = this.status ? this.status.state : "UNAVAILABLE";
    this.button.dataset.wifi = state;
    const label =
      state === "CONNECTED" ? `Wi-Fi: ${this.status.ssid}` : state === "CONNECTING" ? "Wi-Fi: joining" : "Wi-Fi: offline";
    this.button.setAttribute("aria-label", label);
    this.button.title = label;
    return this.status;
  }

  open() {
    clearTimeout(this.closeTimer);
    this.root.hidden = false;
    this.showNetworks();
    this.scan();
  }

  close() {
    clearTimeout(this.closeTimer);
    this.root.hidden = true;
    this.attempt += 1;
    this.ui.password.value = "";
    this.check();
  }

  back() {
    if (this.page === "password") {
      this.showNetworks();
    } else {
      this.close();
    }
  }

  get page() {
    return this.root.dataset.page;
  }

  setPage(page, title) {
    this.root.dataset.page = page;
    this.ui.title.textContent = title;
  }

  // The networks in reach.

  showNetworks() {
    this.setPage("networks", "Let’s get online!");
    this.renderNetworks(this.networks);
  }

  async scan() {
    if (this.scanning) {
      return;
    }
    this.scanning = true;
    this.root.classList.add("scanning");
    this.ui.note.textContent = "Looking for Wi-Fi…";
    try {
      await this.check();
      const response = await fetch("/v1/wifi/scan", { method: "POST" });
      if (response.status === 409) {
        this.ui.note.textContent = "Joining a network. Look again in a moment.";
        return;
      }
      if (!response.ok) {
        this.networks = [];
        this.renderNetworks([]);
        this.ui.note.textContent = "The Wi-Fi isn’t answering. Look again in a moment.";
        return;
      }
      this.networks = (await response.json()).networks;
      this.renderNetworks(this.networks);
      this.ui.note.textContent = this.networks.length ? "" : "No Wi-Fi found. Look again!";
    } catch {
      this.ui.note.textContent = "The service isn’t reachable.";
    } finally {
      this.scanning = false;
      this.root.classList.remove("scanning");
    }
  }

  renderNetworks(networks) {
    const joined = this.status && this.status.state === "CONNECTED" ? this.status.ssid : null;
    this.ui.networks.replaceChildren(
      ...networks.map((network) => {
        const item = document.createElement("li");
        const row = document.createElement("button");
        row.type = "button";
        row.className = "wifi-network" + (network.ssid === joined ? " joined" : "");
        row.innerHTML = `${signalIcon(network.signal)}<span class="wifi-network-name"></span>
          ${network.ssid === joined ? `<span class="wifi-tag">Connected</span>` : ""}
          ${network.secured ? LOCK_ICON : ""}`;
        // Network names are any text, so never markup.
        row.querySelector(".wifi-network-name").textContent = network.ssid;
        row.addEventListener("click", () => this.pick(network));
        item.append(row);
        return item;
      }),
    );
  }

  pick(network) {
    if (network.secured) {
      this.showPassword(network);
    } else {
      this.join(network.ssid, "");
    }
  }

  // The password, and for a network not in the list, its name.

  showPassword(network, error = "") {
    this.network = network;
    const other = network === null;
    this.setPage("password", other ? "Other network" : `Password for ${network.ssid}`);
    this.ui.ssidField.hidden = !other;
    this.root.classList.toggle("other", other);
    if (other && !error) {
      this.ui.ssid.value = "";
    }
    if (!error) {
      this.ui.password.value = "";
    }
    this.setReveal(false);
    this.showError(error);
    this.keyboard.reset();
    this.focus(other && !this.ui.ssid.value ? this.ui.ssid : this.ui.password);
  }

  focus(input) {
    this.focused = input;
    this.ui.ssidField.classList.toggle("focused", input === this.ui.ssid);
    this.ui.passwordField.classList.toggle("focused", input === this.ui.password);
    if (document.activeElement !== input) {
      input.focus({ preventScroll: true });
    }
  }

  type(text) {
    const input = this.focused;
    if (!input || input.value.length + text.length > input.maxLength) {
      return;
    }
    const start = input.selectionStart ?? input.value.length;
    const end = input.selectionEnd ?? input.value.length;
    input.setRangeText(text, start, end, "end");
    input.dispatchEvent(new Event("input"));
    this.focus(input);
  }

  erase(all) {
    const input = this.focused;
    if (!input) {
      return;
    }
    if (all) {
      input.value = "";
    } else {
      let start = input.selectionStart ?? input.value.length;
      const end = input.selectionEnd ?? input.value.length;
      if (start === end && start > 0) {
        start -= 1;
      }
      input.setRangeText("", start, end, "end");
    }
    input.dispatchEvent(new Event("input"));
    this.focus(input);
  }

  toggleReveal() {
    this.setReveal(this.ui.password.type === "password");
    this.focus(this.ui.password);
  }

  setReveal(reveal) {
    this.ui.password.type = reveal ? "text" : "password";
    this.ui.reveal.setAttribute("aria-pressed", String(reveal));
    this.ui.reveal.setAttribute("aria-label", reveal ? "Hide the password" : "Show the password");
  }

  showError(text) {
    this.ui.error.textContent = text;
  }

  submit() {
    if (this.page !== "password") {
      return;
    }
    const other = this.network === null;
    const ssid = other ? this.ui.ssid.value.trim() : this.network.ssid;
    const password = this.ui.password.value;
    if (!ssid) {
      this.showError("What’s the network called?");
      this.focus(this.ui.ssid);
      return;
    }
    if (new TextEncoder().encode(ssid).length > 32) {
      this.showError("That name is too long for a network.");
      this.focus(this.ui.ssid);
      return;
    }
    // A network in the list needs its password; one not listed may be open.
    if (password || !other) {
      if (password.length < 8 || password.length > 63) {
        this.showError("A Wi-Fi password has 8 to 63 letters, numbers or signs.");
        this.focus(this.ui.password);
        return;
      }
    }
    this.join(ssid, password);
  }

  // Joining.

  async join(ssid, password) {
    const attempt = ++this.attempt;
    const network = this.network;
    this.setPage("joining", "Hold on tight!");
    this.ui.joiningText.textContent = `Joining ${ssid}`;
    let result;
    try {
      const response = await fetch("/v1/wifi/connect", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ssid, password }),
      });
      result = response.ok
        ? await response.json()
        : {
            message:
              response.status === 409
                ? "Already joining a network. Try again in a moment."
                : response.status === 422
                  ? "That doesn’t look like a Wi-Fi password."
                  : "The Wi-Fi isn’t answering. Try again in a moment.",
          };
    } catch {
      result = { message: "The service isn’t reachable." };
    }
    if (attempt !== this.attempt || !this.isOpen) {
      return; // The user went on, or closed the sheet.
    }
    if (result.status) {
      this.status = result.status;
    }
    if (result.status && !result.error) {
      this.joined(ssid);
      return;
    }
    const message = result.message ?? WIFI_ERRORS[result.error](ssid);
    if (password || network === null) {
      this.showPassword(network, message);
    } else {
      this.showNetworks();
      this.ui.note.textContent = message;
    }
  }

  joined(ssid) {
    this.setPage("joined", "All set!");
    this.ui.joinedText.textContent = `Connected to ${ssid}`;
    this.check();
    this.closeTimer = setTimeout(() => this.close(), WIFI_JOINED_MS);
  }
}

// The keyboard: letters, numbers and signs, enough for any WPA password,
// which is printable ASCII.

const KEY_LAYOUTS = {
  letters: [
    [..."qwertyuiop"],
    [..."asdfghjkl"],
    ["shift", ..."zxcvbnm", "erase"],
    ["numbers", "space", "enter"],
  ],
  numbers: [
    [..."1234567890"],
    [..."-/:;()$&@\""],
    ["symbols", ...".,?!'", "erase"],
    ["letters", "space", "enter"],
  ],
  symbols: [
    [..."[]{}#%^*+="],
    [..."_\\|~<>`"],
    ["numbers", ...".,?!'", "erase"],
    ["letters", "space", "enter"],
  ],
};

const KEY_LABELS = {
  numbers: "123",
  symbols: "#+=",
  letters: "ABC",
  space: "space",
  enter: "Join",
};

const KEY_ICONS = {
  shift: `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 4l7.5 8H15v7H9v-7H4.5z" /></svg>`,
  erase: `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M9 5h11v14H9l-6-7z" /><path d="M12.5 9.5l5 5M17.5 9.5l-5 5" /></svg>`,
};

// Holding the erase key erases all, after this many ms.
const ERASE_ALL_MS = 700;
// Two taps of shift this close, in ms, keep capitals on.
const CAPS_LOCK_MS = 400;

class OnScreenKeyboard {
  // The element to draw it in, and what it does: type(text),
  // erase(all) and enter().
  constructor(element, actions) {
    this.element = element;
    this.actions = actions;
    this.reset();
    // Keys never take the focus from the input they type into.
    element.addEventListener("mousedown", (event) => event.preventDefault());
    element.addEventListener("pointerdown", (event) => {
      const key = event.target.closest(".key");
      if (key && key.dataset.key === "erase") {
        this.eraseTimer = setTimeout(() => {
          this.erasedAll = true;
          actions.erase(true);
        }, ERASE_ALL_MS);
      }
    });
    const cancel = () => clearTimeout(this.eraseTimer);
    element.addEventListener("pointerup", cancel);
    element.addEventListener("pointercancel", cancel);
    element.addEventListener("pointerleave", cancel);
    element.addEventListener("click", (event) => {
      const key = event.target.closest(".key");
      if (key) {
        this.press(key.dataset.key);
      }
    });
  }

  reset() {
    this.layout = "letters";
    this.shift = "off"; // "off", "once" or "lock".
    this.lastShift = 0;
    this.render();
  }

  press(name) {
    const { actions } = this;
    if (name === "shift") {
      // A tap capitalizes the next letter, a double tap all of them.
      const now = Date.now();
      if (this.shift === "off") {
        this.shift = "once";
      } else if (this.shift === "once" && now - this.lastShift < CAPS_LOCK_MS) {
        this.shift = "lock";
      } else {
        this.shift = "off";
      }
      this.lastShift = now;
      this.render();
    } else if (name === "erase") {
      if (!this.erasedAll) {
        actions.erase(false);
      }
      this.erasedAll = false;
    } else if (name === "enter") {
      actions.enter();
    } else if (name in KEY_LAYOUTS) {
      this.layout = name;
      this.render();
    } else {
      const text = name === "space" ? " " : this.shift !== "off" ? name.toUpperCase() : name;
      actions.type(text);
      if (this.shift === "once") {
        this.shift = "off";
        this.render();
      }
    }
  }

  render() {
    this.element.dataset.shift = this.shift;
    this.element.replaceChildren(
      ...KEY_LAYOUTS[this.layout].map((names) => {
        const row = document.createElement("div");
        row.className = "key-row";
        for (const name of names) {
          const key = document.createElement("button");
          key.type = "button";
          key.tabIndex = -1;
          key.dataset.key = name;
          const single = [...name].length === 1;
          key.className = "key" + (single ? "" : ` key-${name}`);
          if (KEY_ICONS[name]) {
            key.innerHTML = KEY_ICONS[name];
            key.setAttribute("aria-label", name === "shift" ? "Shift" : "Erase");
          } else {
            key.textContent = single ? (this.shift !== "off" ? name.toUpperCase() : name) : KEY_LABELS[name];
          }
          row.append(key);
        }
        return row;
      }),
    );
  }
}
