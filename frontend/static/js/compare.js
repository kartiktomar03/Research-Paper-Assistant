const $ = (id) => document.getElementById(id);
      let currentPaperId = null;
      let topics = [];
      async function requestJson(url, options = {}) {
        const response = await fetch(url, options);
        const data = await response
          .json()
          .catch(() => ({ error: "Unexpected server response" }));
        if (!response.ok)
          throw new Error(data.error || `Request failed (${response.status})`);
        return data;
      }
      // Read POST SSE incrementally. Each event is framed by a blank line; boundaries
      // may occur in the middle of a network chunk or UTF-8 codepoint.
      async function streamJson(url, payload, onDelta, onDone, signal) {
        const response = await fetch(url, {
          method: "POST",
          headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
          body: JSON.stringify(payload), signal,
        });
        if (!response.ok) {
          const d = await response.json().catch(() => ({}));
          throw new Error(d.error || `Request failed (${response.status})`);
        }
        if (!response.body) throw new Error("Streaming is not supported by this browser.");
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        let completed = false;
        function dispatch(frame) {
          let event = "message", data = "";
          for (const line of frame.split("\n")) {
            if (line.startsWith("event:")) event = line.slice(6).trim();
            else if (line.startsWith("data:")) data += line.slice(5).trimStart();
          }
          if (!data) return;
          const value = JSON.parse(data);
          if (event === "delta") onDelta(value.text || "");
          else if (event === "done") { completed = true; onDone?.(value); }
          else if (event === "error") throw new Error(value.error || "Generation failed");
        }
        try {
          while (true) {
            const { value, done } = await reader.read();
            buffer += decoder.decode(value || new Uint8Array(), { stream: !done }).replace(/\r\n/g, "\n");
            let end;
            while ((end = buffer.indexOf("\n\n")) !== -1) {
              const frame = buffer.slice(0, end);
              buffer = buffer.slice(end + 2);
              dispatch(frame);
            }
            if (done) break;
          }
          if (buffer.trim()) dispatch(buffer);
          if (!completed) throw new Error("Stream disconnected before completion.");
        } finally { reader.releaseLock(); }
      }
      let summaryAbort = null;
      async function checkHealth() {
        try {
          const d = await requestJson("/api/health");
          $("health").textContent = `Connected · ${d.vector_store}`;
        } catch (e) {
          $("health").textContent = "Service status unavailable";
        }
      }
      function showStatus(el, msg, isError = false) {
        el.textContent = msg;
        el.classList.toggle("error", isError);
      }
      function escapeHtml(s) {
        return String(s ?? "").replace(
          /[&<>"']/g,
          (c) =>
            ({
              "&": "&amp;",
              "<": "&lt;",
              ">": "&gt;",
              '"': "&quot;",
              "'": "&#39;",
            })[c],
        );
      }
      // Multi-paper selection is independent of the current single-paper dashboard.
      async function refreshPaperChoices() {
        const root = $("paperChoices");
        const checked = new Set([...root.querySelectorAll("input:checked")].map(x => x.value));
        try {
          const data = await requestJson("/api/papers");
          root.replaceChildren();
          if (!data.papers.length) {
            root.textContent = "No papers uploaded yet.";
            return;
          }
          data.papers.forEach((paper) => {
            const label = document.createElement("label");
            label.className = "paper-choice";
            const input = document.createElement("input");
            input.type = "checkbox";
            input.value = paper.paper_id;
            input.checked = checked.has(paper.paper_id);
            const title = document.createElement("span");
            title.textContent = paper.title || paper.filename;
            const pages = document.createElement("small");
            pages.textContent = `${paper.page_count || "?"} pages`;
            const info = document.createElement("div");
            info.className = "paper-choice-details";
            info.append(title, pages);
            const del = document.createElement("button");
            del.type = "button";
            del.className = "btn secondary paper-delete";
            del.textContent = "Delete ×";
            del.setAttribute("aria-label", `Delete ${paper.title || paper.filename}`);
            del.addEventListener("click", async (event) => {
              event.preventDefault();
              event.stopPropagation();
              if (!confirm(`Permanently delete "${paper.title || paper.filename}" and its indexed data?`)) return;
              del.disabled = true;
              try {
                await requestJson(`/api/paper/${encodeURIComponent(paper.paper_id)}`, { method: "DELETE" });
                showStatus($("compareStatus"), "Paper and indexed data deleted.");
                await refreshPaperChoices();
              } catch (err) {
                del.disabled = false;
                showStatus($("compareStatus"), err.message, true);
              }
            });
            label.append(input, info);
            const wrapper = document.createElement("div");
            wrapper.className = "paper-choice-wrap";
            wrapper.append(label, del);
            root.appendChild(wrapper);
          });
        } catch (err) {
          showStatus($("compareStatus"), `Unable to load papers: ${err.message}`, true);
        }
      }
      $("refreshPapers").addEventListener("click", refreshPaperChoices);
      $("compareForm").addEventListener("submit", async (event) => {
        event.preventDefault();
        const selected = [...$("paperChoices").querySelectorAll("input:checked")].map(x => x.value);
        if (selected.length < 2 || selected.length > 5) {
          showStatus($("compareStatus"), "Select 2 to 5 papers first.", true);
          return;
        }
        const message = $("compareQuestion").value.trim();
        if (!message) return;
        $("compareQuestion").value = ""; // Restore placeholder immediately after submit.
        showStatus($("compareStatus"), `Comparing ${selected.length} papers…`);
        $("compareBtn").disabled = true;
        $("compareBtn").textContent = "Comparing…";
        const root = $("compareMessages");
        const question = document.createElement("div");
        question.className = "bubble user";
        question.textContent = message;
        const response = document.createElement("div");
        response.className = "bubble ai streaming";
        const content = document.createElement("div");
        content.className = "bubble-text";
        response.append(content);
        root.append(question, response);
        root.scrollTop = root.scrollHeight;
        try {
          await streamJson("/api/compare/stream", { paper_ids: selected, message },
            (delta) => { content.textContent += delta; root.scrollTop = root.scrollHeight; },
            (done) => {
              const sourceRoot = document.createElement("div");
              sourceRoot.className = "sources";
              (done.sources || []).forEach(source => {
                const tag = document.createElement("span");
                tag.className = "source";
                tag.textContent = `${source.paper_label} · Page ${source.page} · ${source.label}`;
                tag.title = source.paper_title || "";
                sourceRoot.append(tag);
              });
              response.append(sourceRoot);
            });
          showStatus($("compareStatus"), "Comparison complete.");
        } catch (err) {
          content.textContent += `\nError: ${err.message}`;
          showStatus($("compareStatus"), err.message, true);
        } finally {
          response.classList.remove("streaming");
          $("compareBtn").disabled = false;
          $("compareBtn").textContent = "Compare ↗";
        }
      });
      checkHealth();
      refreshPaperChoices();
