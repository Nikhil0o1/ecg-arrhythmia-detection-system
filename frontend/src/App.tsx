import Navbar from "@/components/Navbar";
import Dashboard from "@/pages/Dashboard";
import ToastContainer from "@/components/ToastContainer";

export default function App() {
  return (
    <div className="flex min-h-screen flex-col dark:bg-clinical-bg bg-gray-50 dark:bg-grid-pattern bg-grid bg-grid-pattern transition-colors duration-300">
      <Navbar />
      <main className="flex-1">
        <Dashboard />
      </main>
      <ToastContainer />
    </div>
  );
}
