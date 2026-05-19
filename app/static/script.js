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
            body: JSON.stringify({ question }),
        });

        const data = await response.json();

        if (!response.ok || data.error) {
            showError(data.error || "An unexpected error occurred. Please try again.");
            return;
        }

        // Answer
        document.getElementById("answerText").textContent = data.answer;
        document.getElementById("answerCard").classList.remove("hidden");

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

// Allow submitting with Ctrl+Enter
document.getElementById("questionInput").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && e.ctrlKey) askQuestion();
});
