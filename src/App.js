import React from 'react';
import './App.css';

function App() {
  return (
    <div className="App">
      <header className="App-header">
        {/* Your existing content */}
      </header>
      <main>
        <langflow-chat
          window_title="Vector Store RAG (1)"
          flow_id="07ebf377-ffae-40de-a270-b7f9eefe56cd"
          host_url="http://127.0.0.1:7861"
        ></langflow-chat>
      </main>
    </div>
  );
}

export default App;
