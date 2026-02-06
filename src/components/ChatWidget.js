console.log("CHAT WIDGET LOADED FROM CORRECT PATH");

import React, { useState } from 'react';
import './ChatWidget.css'; // Import the associated CSS file

const ChatWidget = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState([{ text: 'Hello! How can I assist you today?', sender: 'bot' }]);
  const [inputValue, setInputValue] = useState('');

  const toggleChat = () => {
    setIsOpen(!isOpen);
  };

  const handleSendMessage = async () => {
    if (!inputValue.trim()) return;

    console.log("MESSAGE SENDING...");

    // Add user message to the chat
    const userMessage = { text: inputValue, sender: 'user' };
    setMessages(prev => [...prev, userMessage]);
    setInputValue(''); // Clear the input field

    try {
      // Call the backend API
      const response = await fetch('https://mohammadtouqeer-rag-chatbot-backend.hf.space/api/query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query: inputValue }),
      });

      const data = await response.json();

      // Log the full backend response for debugging
      console.log("FULL BACKEND DATA:", data);

      // Handle the response based on new requirements
      let botResponse;
      if (data.answer && data.answer !== "LLM Bypassed") {
        botResponse = data.answer;
      } else if (data.answer?.includes("Bypassed") || !data.answer) {
        botResponse = data.sources?.[0]?.text || "Sorry, I didn't understand that.";
      } else {
        botResponse = data.answer || "Sorry, I didn't understand that.";
      }

      // Add bot response to the chat
      setMessages(prev => [...prev, { text: botResponse, sender: 'bot' }]);
    } catch (error) {
      console.error('Error sending message:', error);
      setMessages(prev => [...prev, { text: 'Error connecting to the server.', sender: 'bot' }]);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter') {
      handleSendMessage();
    }
  };

  return (
    <div className="chat-widget">
      {isOpen ? (
        <div className="chat-window">
          <div className="chat-header">
            <h3>AI Assistant</h3>
            <button className="close-button" onClick={toggleChat}>×</button>
          </div>
          <div className="chat-messages">
            {messages.map((msg, index) => (
              <div key={index} className={`message ${msg.sender}`}>
                {msg.text}
              </div>
            ))}
          </div>
          <div className="chat-input-area">
            <input
              type="text"
              placeholder="Type your message..."
              className="chat-input"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyPress={handleKeyPress}
            />
            <button className="send-button" onClick={handleSendMessage}>Send</button>
          </div>
        </div>
      ) : (
        <button className="chat-toggle-button" onClick={toggleChat}>
          💬
        </button>
      )}
    </div>
  );
};

export default ChatWidget;