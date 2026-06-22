const numberFormat = new Intl.NumberFormat("en");

const paths = {
  assistant: "assistant.yaml",
  manifest: "corpus/manifest.yaml",
  questions: "eval/questions.yaml",
};

function setText(id, value) {
  const element = document.getElementById(id);
  if (element) element.textContent = value;
}

async function readText(path) {
  const response = await fetch(path, { cache: "no-cache" });
  if (!response.ok) throw new Error(`Unable to load ${path}`);
  return response.text();
}

function countMatches(text, pattern) {
  const matches = text.match(pattern);
  return matches ? matches.length : 0;
}

function topEntries(values, limit = 8) {
  const counts = new Map();
  for (const value of values) {
    const key = value.trim();
    if (!key) continue;
    counts.set(key, (counts.get(key) || 0) + 1);
  }
  return [...counts.entries()]
    .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
    .slice(0, limit);
}

function extractSection(text, sectionName, nextSectionNames = []) {
  const start = text.indexOf(`\n${sectionName}:`);
  if (start === -1 && !text.startsWith(`${sectionName}:`)) return "";
  const sectionStart = start === -1 ? 0 : start + 1;
  let sectionEnd = text.length;
  for (const name of nextSectionNames) {
    const index = text.indexOf(`\n${name}:`, sectionStart + sectionName.length + 1);
    if (index !== -1 && index < sectionEnd) sectionEnd = index;
  }
  return text.slice(sectionStart, sectionEnd);
}

function extractListValues(block, key) {
  const values = [];
  const lines = block.split(/\r?\n/);
  let inList = false;
  let indent = 0;
  for (const line of lines) {
    const keyMatch = line.match(new RegExp(`^(\\s*)${key}:\\s*$`));
    if (keyMatch) {
      inList = true;
      indent = keyMatch[1].length;
      continue;
    }
    if (inList) {
      const item = line.match(/^\s*-\s+(.+?)\s*$/);
      if (item) {
        values.push(item[1].replace(/^["']|["']$/g, ""));
        continue;
      }
      const nextKey = line.match(/^(\s*)[A-Za-z0-9_]+:/);
      if (nextKey && nextKey[1].length <= indent) inList = false;
    }
  }
  return values;
}

function parseDocuments(manifestText, limit = 5) {
  const documentsText = extractSection(manifestText, "documents");
  const blocks = documentsText.split(/\n(?=- source_id:)/).filter((block) => /\n\s*title:/.test(block));
  return blocks.slice(0, limit).map((block) => {
    const title = block.match(/\ntitle:\s*(.+)/)?.[1]?.replace(/^["']|["']$/g, "") || "Untitled document";
    const url = block.match(/\n\s*url:\s*(.+)/)?.[1] || "";
    const year = block.match(/\n\s*year:\s*(.+)/)?.[1] || "";
    const language = block.match(/\n\s*language:\s*(.+)/)?.[1] || "";
    return { title, url, year, language };
  });
}

function renderTags(id, entries) {
  const element = document.getElementById(id);
  if (!element) return;
  element.replaceChildren();
  if (!entries.length) {
    const empty = document.createElement("span");
    empty.textContent = "No data";
    element.appendChild(empty);
    return;
  }
  for (const [label, count] of entries) {
    const tag = document.createElement("span");
    tag.textContent = `${label} ${count}`;
    element.appendChild(tag);
  }
}

function renderDocuments(documents) {
  const element = document.getElementById("document-list");
  if (!element) return;
  element.replaceChildren();
  if (!documents.length) {
    const empty = document.createElement("p");
    empty.className = "muted";
    empty.textContent = "No documents found in the manifest.";
    element.appendChild(empty);
    return;
  }
  for (const documentItem of documents) {
    const article = document.createElement("article");
    article.className = "document-card";
    const title = document.createElement("h4");
    if (documentItem.url) {
      const link = document.createElement("a");
      link.href = documentItem.url;
      link.textContent = documentItem.title;
      link.rel = "noreferrer noopener";
      title.appendChild(link);
    } else {
      title.textContent = documentItem.title;
    }
    const meta = document.createElement("p");
    meta.textContent = [documentItem.year, documentItem.language].filter(Boolean).join(" · ");
    article.append(title, meta);
    element.appendChild(article);
  }
}

async function hydratePage() {
  try {
    const [assistantText, manifestText, questionsText] = await Promise.all([
      readText(paths.assistant),
      readText(paths.manifest),
      readText(paths.questions),
    ]);

    const assistantId = assistantText.match(/^assistant_id:\s*(.+)$/m)?.[1]?.trim() || "sgp_ai";
    const documentsText = extractSection(manifestText, "documents");
    const sourceText = extractSection(manifestText, "sources", ["documents"]);
    const documentCount = countMatches(documentsText, /^- source_id:/gm);
    const sourceCount = countMatches(sourceText, /^- source_id:/gm);
    const chunkCount = countMatches(documentsText, /^\s+- section_title:/gm);
    const questionCount = countMatches(questionsText, /^\s+- question_id:/gm);
    const topics = topEntries(extractListValues(documentsText, "topic_tags"), 10);
    const languages = topEntries([...documentsText.matchAll(/^\s*language:\s*(.+)$/gm)].map((match) => match[1]), 8);

    setText("assistant-id", assistantId);
    setText("document-count", numberFormat.format(documentCount));
    setText("chunk-count", numberFormat.format(chunkCount));
    setText("question-count", numberFormat.format(questionCount));
    setText("hero-doc-count", `${numberFormat.format(documentCount)} documents from ${numberFormat.format(sourceCount)} source${sourceCount === 1 ? "" : "s"}`);
    renderTags("topic-list", topics);
    renderTags("language-list", languages);
    renderDocuments(parseDocuments(manifestText));
  } catch (error) {
    setText("hero-doc-count", "Corpus files available");
    setText("document-count", "Open");
    setText("chunk-count", "Open");
    setText("question-count", "Open");
    renderTags("topic-list", []);
    renderTags("language-list", []);
    const list = document.getElementById("document-list");
    if (list) {
      list.innerHTML = '<p class="muted">Open the manifest directly to inspect documents.</p>';
    }
  }
}

hydratePage();
