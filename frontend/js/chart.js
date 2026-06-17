/**
 * chart.js — Donut chart creation and updates using Chart.js.
 */

import { EMOTION_CONFIG } from "./config.js";

let donutChart = null;

/**
 * Create or update the donut chart with the latest probabilities.
 * @param {Object} probabilities - Emotion→probability mapping.
 * @param {string} activeEmotion - The predicted emotion key.
 */
export function updateDonut(probabilities, activeEmotion) {
  const labels  = Object.keys(probabilities);
  const values  = Object.values(probabilities);
  const colors  = labels.map((em) =>
    em === activeEmotion
      ? (EMOTION_CONFIG[em] ?? EMOTION_CONFIG.neutral).color
      : "rgba(150,160,190,0.18)"
  );
  const borders = labels.map((em) =>
    em === activeEmotion
      ? (EMOTION_CONFIG[em] ?? EMOTION_CONFIG.neutral).color
      : "rgba(150,160,190,0.08)"
  );

  if (donutChart) {
    donutChart.data.datasets[0].data            = values;
    donutChart.data.datasets[0].backgroundColor = colors;
    donutChart.data.datasets[0].borderColor      = borders;
    donutChart.update("active");
  } else {
    const ctx = document.getElementById("donutChart").getContext("2d");
    donutChart = new Chart(ctx, {
      type: "doughnut",
      data: {
        labels,
        datasets: [{
          data: values,
          backgroundColor: colors,
          borderColor:     borders,
          borderWidth:     2,
          hoverOffset:     6,
        }],
      },
      options: {
        cutout:    "68%",
        animation: { duration: 700, easing: "easeInOutCubic" },
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (ctx) => ` ${(ctx.raw * 100).toFixed(1)}%`,
            },
          },
        },
      },
    });
  }
}
