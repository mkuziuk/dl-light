const sectionOrder = [
  "Главная идея",
  "Минимум для ответа",
  "Формулы / схема",
  "Диаграмма",
  "Уточнения экзаменатора",
  "Частые ошибки",
];

const ratingLabels = {
  1: "Lost",
  2: "Shaky",
  3: "Workable",
  4: "Solid",
  5: "Exam-ready",
};

const state = {
  dashboard: null,
  questions: [],
  queue: [],
  current: null,
  revealed: false,
  view: "dashboard",
};

if (window.mermaid) {
  window.mermaid.initialize({
    startOnLoad: false,
    securityLevel: "strict",
    theme: "base",
    themeVariables: {
      background: "#ffffff",
      primaryColor: "#f7f4ed",
      primaryTextColor: "#181b1f",
      primaryBorderColor: "#d8d2c6",
      lineColor: "#66706f",
      secondaryColor: "#eef5f2",
      tertiaryColor: "#ffffff",
      fontFamily:
        'Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
    },
  });
}

document.addEventListener("DOMContentLoaded", async () => {
  bindNavigation();
  bindControls();
  await loadData();
  renderAll();
});

async function loadData() {
  const [dashboard, questions] = await Promise.all([
    getJSON("/api/dashboard"),
    getJSON("/api/questions"),
  ]);
  state.dashboard = dashboard;
  state.questions = questions;
}

function bindNavigation() {
  document.querySelectorAll("[data-view]").forEach((button) => {
    button.addEventListener("click", () => setView(button.dataset.view));
  });
}

function bindControls() {
  document.querySelectorAll("[data-session]").forEach((button) => {
    button.addEventListener("click", () => startSession(button.dataset.session));
  });

  document.getElementById("next-button").addEventListener("click", selectNextQuestion);
  document.getElementById("shuffle-button").addEventListener("click", () => {
    state.queue = shuffle([...state.queue]);
    renderReviewQueue();
    if (state.queue.length) {
      selectQuestion(state.queue[0]);
    }
  });

  ["search-input", "topic-filter", "rating-filter"].forEach((id) => {
    document.getElementById(id).addEventListener("input", renderQuestionTable);
  });
}

function renderAll() {
  renderDashboard();
  renderTopicFilter();
  renderQuestionTable();
  if (!state.queue.length && state.dashboard?.suggested_queue?.length) {
    state.queue = state.dashboard.suggested_queue.map((question) => question.number);
  }
  renderReviewQueue();
  renderQuestionPanel();
}

function setView(viewName) {
  state.view = viewName;
  document.querySelectorAll(".view").forEach((view) => {
    view.classList.toggle("is-active", view.id === `${viewName}-view`);
  });
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.classList.toggle("is-active", tab.dataset.view === viewName);
  });
}

function renderDashboard() {
  const summary = state.dashboard.summary;
  const average = summary.average_rating === null ? "n/a" : summary.average_rating;
  document.getElementById("summary").innerHTML = [
    metric(summary.total_questions, "Exam questions"),
    metric(summary.rated_questions, "Rated"),
    metric(summary.unrated_questions, "Unrated"),
    metric(average, "Average rating"),
  ].join("");

  document.getElementById("suggested-queue").innerHTML = renderQuestionList(
    state.dashboard.suggested_queue,
  );
  document.getElementById("weakest-list").innerHTML = renderQuestionList(
    state.dashboard.weakest_questions.slice(0, 8),
  );
  document.getElementById("coverage-note").textContent =
    `${summary.rated_questions}/${summary.total_questions} rated`;
  renderTopicCoverage();
}

function metric(value, label) {
  return `
    <div class="metric">
      <span class="metric-value">${escapeHTML(String(value))}</span>
      <span class="metric-label">${escapeHTML(label)}</span>
    </div>
  `;
}

function renderQuestionList(questions) {
  if (!questions.length) {
    return `<div class="empty">No questions to show.</div>`;
  }

  return questions
    .map(
      (question) => `
        <button class="question-item" data-question="${question.number}">
          <span class="question-number">${padNumber(question.number)}</span>
          <span>
            <span class="question-title">${escapeHTML(question.title)}</span>
            <span class="question-topic">${escapeHTML(shortTopic(question.topic))}</span>
          </span>
          ${ratingPill(question.progress?.rating)}
        </button>
      `,
    )
    .join("");
}

function renderTopicCoverage() {
  const rows = state.dashboard.topic_coverage
    .map((topic) => {
      const average = topic.average_rating === null ? "n/a" : topic.average_rating;
      return `
        <div class="topic-row">
          <div>
            <strong>${escapeHTML(shortTopic(topic.topic))}</strong>
            <div class="subtle">${topic.rated}/${topic.total} rated, ${topic.weak_count} weak or unrated</div>
          </div>
          <div class="progress-track" aria-label="${topic.coverage_percent}% covered">
            <div class="progress-fill" style="width: ${topic.coverage_percent}%"></div>
          </div>
          <div>${topic.coverage_percent}%</div>
          <div>${average}</div>
        </div>
      `;
    })
    .join("");
  document.getElementById("topic-coverage").innerHTML = rows;
}

function startSession(mode) {
  let queue = [];
  if (mode === "balanced") {
    queue = state.dashboard.suggested_queue.map((question) => question.number);
  } else if (mode === "weakest") {
    queue = state.dashboard.weakest_questions.map((question) => question.number);
  } else if (mode === "unrated") {
    queue = state.questions
      .filter((question) => !question.progress?.rating)
      .map((question) => question.number);
  } else if (mode === "random") {
    queue = shuffle(state.questions.map((question) => question.number)).slice(0, 12);
  }

  state.queue = queue;
  renderReviewQueue();
  setView("review");
  if (queue.length) {
    selectQuestion(queue[0]);
  } else {
    state.current = null;
    renderQuestionPanel("No questions match this session.");
  }
}

async function selectQuestion(number) {
  state.current = await getJSON(`/api/questions/${number}`);
  state.revealed = false;
  renderReviewQueue();
  renderQuestionPanel();
  renderAnswerPanel();
}

function selectNextQuestion() {
  if (!state.queue.length) {
    return;
  }
  const currentNumber = state.current?.number;
  const index = state.queue.indexOf(currentNumber);
  const next = state.queue[index + 1] ?? state.queue[0];
  selectQuestion(next);
}

function renderReviewQueue() {
  const list = document.getElementById("review-queue");
  if (!state.queue.length) {
    list.innerHTML = `<div class="empty">Start a session from the dashboard.</div>`;
    return;
  }
  list.innerHTML = state.queue
    .map((number) => {
      const question = state.questions.find((item) => item.number === number);
      if (!question) {
        return "";
      }
      const active = state.current?.number === number ? " is-active" : "";
      return `
        <button class="rail-item${active}" data-question="${number}">
          <span class="rail-number">${padNumber(number)}</span>
          <span>
            <span class="rail-title">${escapeHTML(question.title)}</span>
            <span class="rail-meta">${ratingText(question.progress?.rating)}</span>
          </span>
        </button>
      `;
    })
    .join("");
}

function renderQuestionPanel(message = "") {
  const panel = document.getElementById("question-panel");
  if (message) {
    panel.innerHTML = `<p class="empty">${escapeHTML(message)}</p>`;
    document.getElementById("answer-panel").classList.add("is-hidden");
    return;
  }
  if (!state.current) {
    panel.innerHTML = `<p class="empty">Choose a review queue or open a question.</p>`;
    return;
  }

  const question = state.current;
  const rating = question.progress?.rating ?? null;
  panel.innerHTML = `
    <div class="review-meta">
      <span class="tag">Question ${padNumber(question.number)}</span>
      <span class="tag">${escapeHTML(shortTopic(question.topic))}</span>
      <span class="tag">${escapeHTML(ratingText(rating))}</span>
    </div>
    <h2>${escapeHTML(question.title)}</h2>
    <p class="question-text">${escapeHTML(question.original_question)}</p>
    <div class="actions">
      <button class="primary" id="reveal-button">${state.revealed ? "Hide answer" : "Reveal answer"}</button>
      <button class="ghost" id="open-note-button">Open in Obsidian path</button>
    </div>
    <div class="rating-control" aria-label="Confidence rating">
      ${[1, 2, 3, 4, 5].map((value) => ratingButton(value, rating)).join("")}
      <span class="rating-help">1 = lost, 3 = workable, 5 = exam-ready</span>
    </div>
  `;

  document.getElementById("reveal-button").addEventListener("click", () => {
    state.revealed = !state.revealed;
    renderQuestionPanel();
    renderAnswerPanel();
  });
  document.getElementById("open-note-button").addEventListener("click", () => {
    showToast(question.path);
  });
  panel.querySelectorAll(".rating-button").forEach((button) => {
    button.addEventListener("click", () => saveRating(Number(button.dataset.rating)));
  });
}

function ratingButton(value, currentRating) {
  const selected = value === currentRating ? " is-selected" : "";
  return `
    <button
      class="rating-button${selected}"
      data-rating="${value}"
      title="${ratingLabels[value]}"
      aria-label="Rate ${value}: ${ratingLabels[value]}"
    >${value}</button>
  `;
}

function renderAnswerPanel() {
  const panel = document.getElementById("answer-panel");
  if (!state.current || !state.revealed) {
    panel.classList.add("is-hidden");
    panel.innerHTML = "";
    return;
  }

  const sections = sectionOrder
    .filter((sectionName) => state.current.sections?.[sectionName])
    .map(
      (sectionName) => `
        <section class="answer-section">
          <h3>${escapeHTML(sectionName)}</h3>
          ${renderMarkdown(state.current.sections[sectionName])}
        </section>
      `,
    )
    .join("");

  panel.innerHTML = sections;
  panel.classList.remove("is-hidden");
  renderRichContent(panel);
}

async function saveRating(rating) {
  if (!state.current) {
    return;
  }
  await postJSON(`/api/questions/${state.current.number}/rating`, { rating });
  showToast(`Saved rating ${rating}: ${ratingLabels[rating]}`);
  await loadData();
  state.current = await getJSON(`/api/questions/${state.current.number}`);
  renderAll();
  setView("review");
}

function renderTopicFilter() {
  const topics = [...new Set(state.questions.map((question) => question.topic))].sort();
  const select = document.getElementById("topic-filter");
  select.innerHTML = [
    `<option value="all">All topics</option>`,
    ...topics.map((topic) => `<option value="${escapeAttribute(topic)}">${escapeHTML(shortTopic(topic))}</option>`),
  ].join("");
}

function renderQuestionTable() {
  const search = document.getElementById("search-input").value.trim().toLowerCase();
  const topic = document.getElementById("topic-filter").value;
  const ratingFilter = document.getElementById("rating-filter").value;

  const rows = state.questions.filter((question) => {
    const rating = question.progress?.rating;
    const searchable = `${question.title} ${question.topic} ${question.original_question}`.toLowerCase();
    if (search && !searchable.includes(search)) {
      return false;
    }
    if (topic !== "all" && question.topic !== topic) {
      return false;
    }
    if (ratingFilter === "unrated" && rating) {
      return false;
    }
    if (!["all", "unrated"].includes(ratingFilter) && String(rating) !== ratingFilter) {
      return false;
    }
    return true;
  });

  const table = document.getElementById("question-table");
  table.innerHTML = `
    <div class="question-row table-head">
      <span>No.</span>
      <span>Question</span>
      <span>Topic</span>
      <span>Rating</span>
      <span>Action</span>
    </div>
    ${
      rows.length
        ? rows.map(renderQuestionRow).join("")
        : `<div class="empty">No matching questions.</div>`
    }
  `;
}

function renderQuestionRow(question) {
  return `
    <div class="question-row">
      <span class="question-number">${padNumber(question.number)}</span>
      <span>
        <span class="question-title">${escapeHTML(question.title)}</span>
        <span class="question-topic">${escapeHTML(question.original_question)}</span>
      </span>
      <span>${escapeHTML(shortTopic(question.topic))}</span>
      <span>${ratingPill(question.progress?.rating)}</span>
      <button data-question="${question.number}">Review</button>
    </div>
  `;
}

document.addEventListener("click", (event) => {
  const button = event.target.closest("[data-question]");
  if (!button) {
    return;
  }
  const number = Number(button.dataset.question);
  if (!state.queue.includes(number)) {
    state.queue = [number, ...state.queue.filter((item) => item !== number)];
  }
  setView("review");
  selectQuestion(number);
});

function renderMarkdown(markdown) {
  const lines = markdown.trim().split(/\r?\n/);
  const html = [];
  let listOpen = false;

  const closeList = () => {
    if (listOpen) {
      html.push("</ul>");
      listOpen = false;
    }
  };

  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index];
    const trimmed = line.trim();

    if (!trimmed) {
      closeList();
      continue;
    }

    if (trimmed.startsWith("```")) {
      closeList();
      const language = trimmed.slice(3).trim();
      const block = [];
      index += 1;
      while (index < lines.length && !lines[index].trim().startsWith("```")) {
        block.push(lines[index]);
        index += 1;
      }
      if (language === "mermaid") {
        html.push(`<div class="mermaid diagram-block">${escapeHTML(block.join("\n"))}</div>`);
      } else {
        html.push(`<pre class="code-block"><code>${escapeHTML(block.join("\n"))}</code></pre>`);
      }
      continue;
    }

    if (trimmed === "$$") {
      closeList();
      const block = [];
      index += 1;
      while (index < lines.length && lines[index].trim() !== "$$") {
        block.push(lines[index]);
        index += 1;
      }
      html.push(`<div class="math-block">$$\n${escapeHTML(block.join("\n"))}\n$$</div>`);
      continue;
    }

    if (trimmed.startsWith("![")) {
      closeList();
      html.push(renderImage(trimmed));
      continue;
    }

    if (trimmed.startsWith("- ")) {
      if (!listOpen) {
        html.push("<ul>");
        listOpen = true;
      }
      html.push(`<li>${renderInline(trimmed.slice(2))}</li>`);
      continue;
    }

    closeList();
    if (trimmed.startsWith(">")) {
      html.push(`<blockquote>${renderInline(trimmed.slice(1).trim())}</blockquote>`);
    } else {
      html.push(`<p>${renderInline(trimmed)}</p>`);
    }
  }
  closeList();
  return html.join("");
}

function renderInline(value) {
  return escapeHTML(value)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
}

function renderRichContent(container) {
  renderDiagrams(container);
  renderMath(container);
}

async function renderMath(container) {
  if (!window.MathJax?.typesetPromise) {
    return;
  }
  try {
    await window.MathJax.typesetPromise([container]);
  } catch (error) {
    console.error("MathJax rendering failed", error);
    showToast("Formula rendering failed");
  }
}

async function renderDiagrams(container) {
  if (!window.mermaid?.run) {
    return;
  }
  const diagrams = container.querySelectorAll(".mermaid:not([data-processed])");
  if (!diagrams.length) {
    return;
  }
  try {
    await window.mermaid.run({
      querySelector: "#answer-panel .mermaid:not([data-processed])",
    });
  } catch (error) {
    console.error("Mermaid rendering failed", error);
    diagrams.forEach((diagram) => {
      diagram.classList.add("diagram-error");
      diagram.textContent = "Diagram rendering failed.";
    });
    showToast("Diagram rendering failed");
  }
}

function renderImage(line) {
  const match = line.match(/!\[([^\]]*)]\(([^)]+)\)/);
  if (!match) {
    return `<p>${renderInline(line)}</p>`;
  }
  const alt = match[1] || "Note image";
  const src = normalizeImagePath(match[2]);
  return `
    <figure>
      <img class="note-image" src="${escapeAttribute(src)}" alt="${escapeAttribute(alt)}" loading="lazy" />
      <figcaption class="subtle">${escapeHTML(alt)}</figcaption>
    </figure>
  `;
}

function normalizeImagePath(rawPath) {
  let path = rawPath.trim();
  if (path.startsWith("<") && path.endsWith(">")) {
    path = path.slice(1, -1);
  }
  path = path.replace(/^\.\.\//, "");
  if (path.startsWith("assets/")) {
    return `/${path.split("/").map(encodeURIComponent).join("/")}`;
  }
  return path;
}

function ratingPill(rating) {
  if (!rating) {
    return `<span class="rating-pill">Unrated</span>`;
  }
  const tone = rating >= 4 ? " good" : rating <= 2 ? " weak" : "";
  return `<span class="rating-pill${tone}">${rating}/5</span>`;
}

function ratingText(rating) {
  return rating ? `${rating}/5 ${ratingLabels[rating]}` : "Unrated";
}

function shortTopic(topic) {
  return topic.replace(/^Topic \d+ - /, "");
}

function padNumber(number) {
  return String(number).padStart(2, "0");
}

function shuffle(items) {
  for (let index = items.length - 1; index > 0; index -= 1) {
    const swapIndex = Math.floor(Math.random() * (index + 1));
    [items[index], items[swapIndex]] = [items[swapIndex], items[index]];
  }
  return items;
}

async function getJSON(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`GET ${url} failed`);
  }
  return response.json();
}

async function postJSON(url, body) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    throw new Error(`POST ${url} failed`);
  }
  return response.json();
}

function showToast(message) {
  const toast = document.getElementById("toast");
  toast.textContent = message;
  toast.classList.add("is-visible");
  clearTimeout(showToast.timer);
  showToast.timer = setTimeout(() => toast.classList.remove("is-visible"), 2200);
}

function escapeHTML(value) {
  return value.replace(/[&<>"']/g, (char) => {
    const replacements = {
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      '"': "&quot;",
      "'": "&#039;",
    };
    return replacements[char];
  });
}

function escapeAttribute(value) {
  return escapeHTML(value);
}
