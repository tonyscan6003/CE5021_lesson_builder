// Lesson page behaviour: KaTeX, interactive figure toggles, self-check quizzes.
document.addEventListener("DOMContentLoaded", function () {
  if (window.renderMathInElement) {
    renderMathInElement(document.body, {
      delimiters: [
        { left: "$$", right: "$$", display: true },
        { left: "$", right: "$", display: false },
      ],
      throwOnError: false,
    });
  }

  document.querySelectorAll(".fig-toggle").forEach(function (btn) {
    btn.addEventListener("click", function () {
      var fig = document.getElementById(btn.dataset.target);
      var box = fig.querySelector(".fig-interactive");
      var show = box.hidden;
      box.hidden = !show;
      fig.classList.toggle("show-interactive", show);
      btn.textContent = show ? "Static version" : "Interactive version";
      if (show && window.Plotly) {
        box.querySelectorAll(".plotly-graph-div").forEach(function (g) { Plotly.Plots.resize(g); });
      }
    });
  });

  document.querySelectorAll(".selfcheck").forEach(function (sec) {
    var btn = sec.querySelector(".sc-check");
    btn.addEventListener("click", function () {
      var qs = sec.querySelectorAll(".sc-q"), score = 0, answered = 0;
      qs.forEach(function (q) {
        var sel = q.querySelector("input:checked");
        var fb = q.querySelector(".sc-feedback"), ex = q.querySelector(".sc-explain");
        q.classList.remove("correct", "wrong");
        if (!sel) { fb.hidden = true; ex.hidden = true; return; }
        answered++;
        var ok = sel.value === q.dataset.answer;
        if (ok) score++;
        q.classList.add(ok ? "correct" : "wrong");
        fb.textContent = ok ? "Correct." : "Not quite.";
        fb.hidden = false; ex.hidden = false;
      });
      sec.querySelector(".sc-score").textContent =
        answered ? score + " / " + qs.length + " correct" : "Select an answer first.";
    });
  });
});
