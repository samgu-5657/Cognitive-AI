import "beercss";
import "material-dynamic-colors";
import "./styles.css";
import { createRoot } from "react-dom/client";
import App from "./App.jsx";

const root = createRoot(document.getElementById("root"));
root.render(<App />);
