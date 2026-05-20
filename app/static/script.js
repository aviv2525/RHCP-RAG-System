const conversationHistory = [];

function setLoading(on) {
    const btn = document.getElementById("askBtn");
    document.getElementById("loading").classList.toggle("hidden", !on);
    btn.disabled = on;
    btn.textContent = on ? "Searching…" : "Ask";
}

function showError(message) {
    const card = document.getElementById("errorCard");
    document.getElementById("errorText").textContent = message;
    card.classList.remove("hidden");
}

function clearResults() {
    document.getElementById("errorCard").classList.add("hidden");
    document.getElementById("answerCard").classList.add("hidden");
    document.getElementById("sourcesCard").classList.add("hidden");
    document.getElementById("answerText").textContent = "";
    document.getElementById("sourcesList").innerHTML = "";
}

async function askQuestion() {
    const question = document.getElementById("questionInput").value.trim();

    if (!question) {
        showError("Please enter a question before submitting.");
        return;
    }

    clearResults();
    setLoading(true);

    try {
        const response = await fetch("/ask", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question, history: conversationHistory }),
        });

        const data = await response.json();

        if (!response.ok || data.error) {
            showError(data.error || "An unexpected error occurred. Please try again.");
            return;
        }

        // Answer
        document.getElementById("answerText").textContent = data.answer;
        document.getElementById("answerCard").classList.remove("hidden");

        // Save to history and update conversation display
        conversationHistory.push({ question, answer: data.answer });
        appendToHistory(question, data.answer);

        // Sources
        if (data.sources && data.sources.length > 0) {
            const list = document.getElementById("sourcesList");
            data.sources.forEach((src) => {
                const div = document.createElement("div");
                div.className = "source-chunk";
                div.textContent = src;
                list.appendChild(div);
            });
            document.getElementById("sourcesCard").classList.remove("hidden");
        }

    } catch (err) {
        showError("Could not reach the server. Make sure the app is running.");
    } finally {
        setLoading(false);
    }
}

function appendToHistory(question, answer) {
    const card = document.getElementById("historyCard");
    const list = document.getElementById("historyList");

    // Only show history once there's more than one turn
    if (conversationHistory.length < 2) return;

    card.classList.remove("hidden");

    // Show all turns except the last (which is shown in answerCard)
    list.innerHTML = "";
    conversationHistory.slice(0, -1).forEach((turn) => {
        const item = document.createElement("div");
        item.className = "history-item";
        item.innerHTML = `
            <div class="history-q">Q: ${turn.question}</div>
            <div class="history-a">A: ${turn.answer}</div>
        `;
        list.appendChild(item);
    });
}

// Allow submitting with Ctrl+Enter
document.getElementById("questionInput").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && e.ctrlKey) askQuestion();
});

// Show chosen filename next to file input
document.getElementById("fileInput").addEventListener("change", (e) => {
    const name = e.target.files[0]?.name || "No file chosen";
    document.getElementById("fileName").textContent = name;
});

async function uploadFile() {
    const fileInput = document.getElementById("fileInput");
    const file = fileInput.files[0];

    document.getElementById("uploadSuccess").classList.add("hidden");
    document.getElementById("uploadError").classList.add("hidden");

    if (!file) {
        document.getElementById("uploadErrorText").textContent = "Please choose a file first.";
        document.getElementById("uploadError").classList.remove("hidden");
        return;
    }

    const btn = document.getElementById("uploadBtn");
    btn.disabled = true;
    btn.textContent = "Uploading…";
    document.getElementById("uploadLoading").classList.remove("hidden");

    try {
        const form = new FormData();
        form.append("file", file);

        const response = await fetch("/upload", { method: "POST", body: form });
        const data = await response.json();

        if (!response.ok || data.error) {
            document.getElementById("uploadErrorText").textContent = data.error || "Upload failed.";
            document.getElementById("uploadError").classList.remove("hidden");
        } else {
            document.getElementById("uploadSuccess").textContent = "✓ " + data.message;
            document.getElementById("uploadSuccess").classList.remove("hidden");
            fileInput.value = "";
            document.getElementById("fileName").textContent = "No file chosen";
        }
    } catch (err) {
        document.getElementById("uploadErrorText").textContent = "Could not reach the server.";
        document.getElementById("uploadError").classList.remove("hidden");
    } finally {
        btn.disabled = false;
        btn.textContent = "Upload";
        document.getElementById("uploadLoading").classList.add("hidden");
    }
}
