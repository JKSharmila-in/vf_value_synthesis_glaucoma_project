// ---------------------------------------
// app.js - Frontend Logic
// ---------------------------------------

document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("predictionForm");
  const fileInput = document.getElementById("fundus_image");
  const statusText = document.getElementById("statusText");

  if (!form) return;

  form.addEventListener("submit", async (e) => {
    e.preventDefault();

    if (!fileInput.files.length) {
      alert("Please upload a fundus image");
      return;
    }

    const formData = new FormData();
    formData.append("fundus_image", fileInput.files[0]);

    statusText.innerText = "Predicting VF values... ⏳";

    try {
      const response = await fetch("/predict", {
        method: "POST",
        body: formData
      });

      const data = await response.json();

      if (!data.success) {
        throw new Error(data.error || "Prediction failed");
      }

      // Save VF values in sessionStorage
      sessionStorage.setItem(
        "vf_values",
        JSON.stringify(data.vf_values)
      );

      // Redirect to result page
      window.location.href = "/prediction-result";

    } catch (error) {
      console.error(error);
      statusText.innerText = "Prediction failed ❌";
      alert("Error: " + error.message);
    }
  });
});
