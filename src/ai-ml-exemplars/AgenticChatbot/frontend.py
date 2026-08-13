"""
React frontend for Agentic Chatbot
This file contains the basic structure and components for the frontend
"""
# This is a Python file that will be used to document the React frontend structure
# The actual React code would be in a separate folder structure

FRONTEND_STRUCTURE = """
Agentic Chatbot Frontend Structure
==================================

src/
├── components/
│   ├── ChatInterface.jsx
│   ├── Message.jsx
│   ├── InputArea.jsx
│   └── ToolPanel.jsx
├── services/
│   └── chatService.js
├── App.jsx
└── index.js

components/ChatInterface.jsx:
import React, { useState, useEffect } from 'react';
import Message from './Message';
import InputArea from './InputArea';

const ChatInterface = () => {
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);

  const sendMessage = async (text) => {
    setIsLoading(true);

    try {
      const response = await fetch('http://localhost:8000/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          messages: [{ role: 'user', content: text }],
          user_id: 'default'
        })
      });

      const data = await response.json();

      setMessages(prev => [...prev,
        { role: 'user', content: text },
        { role: 'assistant', content: data.response }
      ]);
    } catch (error) {
      console.error('Error:', error);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="chat-container">
      <div className="messages-area">
        {messages.map((msg, index) => (
          <Message key={index} message={msg} />
        ))}
      </div>
      <InputArea onSendMessage={sendMessage} isLoading={isLoading} />
    </div>
  );
};

export default ChatInterface;

components/Message.jsx:
import React from 'react';

const Message = ({ message }) => {
  return (
    <div className={`message ${message.role}`}>
      <div className="content">
        {message.content}
      </div>
    </div>
  );
};

export default Message;

components/InputArea.jsx:
import React, { useState } from 'react';

const InputArea = ({ onSendMessage, isLoading }) => {
  const [inputValue, setInputValue] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (inputValue.trim() && !isLoading) {
      onSendMessage(inputValue);
      setInputValue('');
    }
  };

  return (
    <form className="input-area" onSubmit={handleSubmit}>
      <input
        type="text"
        value={inputValue}
        onChange={(e) => setInputValue(e.target.value)}
        placeholder="Type your message..."
        disabled={isLoading}
      />
      <button type="submit" disabled={isLoading || !inputValue.trim()}>
        {isLoading ? 'Sending...' : 'Send'}
      </button>
    </form>
  );
};

export default InputArea;

services/chatService.js:
// Service to handle chat API calls
const API_BASE_URL = 'http://localhost:8000';

export const sendMessage = async (messages, userId = 'default') => {
  try {
    const response = await fetch(`${API_BASE_URL}/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        messages,
        user_id: userId
      })
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Error sending message:', error);
    throw error;
  }
};

App.jsx:
import React from 'react';
import ChatInterface from './components/ChatInterface';
import './App.css';

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <h1>Agentic Chatbot</h1>
      </header>
      <main>
        <ChatInterface />
      </main>
    </div>
  );
}

export default App;

index.js:
import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import './index.css';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
"""

print("Frontend structure documentation created.")
print(FRONTEND_STRUCTURE)