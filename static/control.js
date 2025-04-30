// Fade-in observer for smooth transitions
document.addEventListener("DOMContentLoaded", () => {
  const fadeElements = document.querySelectorAll(".fade-in");

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.style.opacity = 1;
          entry.target.style.transform = "translateY(0)";
          observer.unobserve(entry.target);
        }
      });
    },
    {
      threshold: 0.1,
    }
  );

  fadeElements.forEach(el => {
    observer.observe(el);
  });
});

// Highlight rows on hover (optional, for touch devices compatibility)
document.addEventListener("DOMContentLoaded", () => {
  const rows = document.querySelectorAll(".performance-row");
  rows.forEach(row => {
    row.addEventListener("mouseenter", () => {
      row.style.backgroundColor = "#e3f2fd";
    });

    row.addEventListener("mouseleave", () => {
      row.style.backgroundColor = "";
    });
  });
});