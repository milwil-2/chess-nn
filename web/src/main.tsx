import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import "@fontsource/spectral/400.css";
import "@fontsource/spectral/400-italic.css";
import "@fontsource/spectral/500.css";
import "@fontsource/spectral/500-italic.css";
import "@fontsource/spectral/600.css";
import "@fontsource-variable/jetbrains-mono";
import "./styles/global.css";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
