import "./styles/App.css";
import Pages from "./pages/Pages";
import { BrowserRouter } from "react-router-dom";

function App({ className }) {
  return (
    <BrowserRouter>
      <Pages className={className} />
    </BrowserRouter>
  );
}

export default App;
