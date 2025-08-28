function checkSpam() {
    const emailText = document.getElementById("emailInput").value;
    const resultElement = document.getElementById("output");

    if (!emailText.trim()) {
        resultElement.textContent = "Please enter an email!";
        return;
    }

    // For now, simulate spam detection (replace with your ML model later)
    const isSpam = Math.random() > 0.5; // Randomly assigns spam/ham

    if (isSpam) {
        resultElement.innerHTML = "Result: <span class='spam'>Spam! 🚨</span>";
    } else {
        resultElement.innerHTML = "Result: <span class='ham'>Ham (Not Spam)! ✅</span>";
    }
}
