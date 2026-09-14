import { useState } from "react";

const ROLE_OPTIONS = ["student", "faculty", "admin"];

function App() {
  const [question, setQuestion] = useState("");
  const [role, setRole] = useState("student");
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [errorMessage, setErrorMessage] = useState(null);

  async function handleSubmit(event) {
    event.preventDefault();
    if (!question.trim()) {
      return;
    }

    setIsLoading(true);
    setErrorMessage(null);
    setResult(null);

    try {
      const response = await fetch("/query", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-User-Role": role,
        },
        body: JSON.stringify({ question }),
      });

      if (!response.ok) {
        const errorBody = await response.json().catch(() => null);
        throw new Error(errorBody?.detail || `Request failed with status ${response.status}`);
      }

      const data = await response.json();
      setResult(data);
    } catch (error) {
      setErrorMessage(error.message);
    } finally {
      setIsLoading(false);
    }
  }

  const sourceDocumentTitles = result
    ? [...new Set(result.results.map((item) => item.document_title))]
    : [];

  return (
    <div className="page">
      <div className="card">
        <h1>Campus Knowledge Assistant</h1>
        <p className="subtitle">Ask a question about SJSU policies and the MSADI program.</p>

        <form onSubmit={handleSubmit} className="query-form">
          <label htmlFor="role-select">Role</label>
          <select id="role-select" value={role} onChange={(event) => setRole(event.target.value)}>
            {ROLE_OPTIONS.map((roleOption) => (
              <option key={roleOption} value={roleOption}>
                {roleOption}
              </option>
            ))}
          </select>

          <label htmlFor="question-input">Question</label>
          <textarea
            id="question-input"
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            placeholder="e.g. What GPA is required for advancement to candidacy?"
            rows={3}
          />

          <button type="submit" disabled={isLoading || !question.trim()}>
            {isLoading ? "Asking..." : "Ask"}
          </button>
        </form>

        {errorMessage && (
          <div className="panel panel-error">
            <strong>Request failed.</strong> {errorMessage}
          </div>
        )}

        {result && !result.has_sufficient_information && (
          <div className="panel panel-refused">
            <strong>No answer available</strong>
            <p>{result.answer}</p>
          </div>
        )}

        {result && result.has_sufficient_information && (
          <div className="panel panel-answer">
            <h2>Answer</h2>
            <p className="answer-text">{result.answer}</p>

            {sourceDocumentTitles.length > 0 && (
              <div className="section">
                <h3>Sources</h3>
                <ul>
                  {sourceDocumentTitles.map((title) => (
                    <li key={title}>{title}</li>
                  ))}
                </ul>
              </div>
            )}

            {result.staleness_notes.length > 0 && (
              <div className="section">
                <h3>Notes</h3>
                <ul>
                  {result.staleness_notes.map((note, index) => (
                    <li key={index}>{note}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
