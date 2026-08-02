import { useEffect, useRef, useState, useCallback } from 'react';

export const useWebSocket = (url) => {
  const ws = useRef(null);
  const [isConnected, setIsConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    try {
      ws.current = new WebSocket(url);

      ws.current.onopen = () => {
        setIsConnected(true);
        setError(null);
      };

      ws.current.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          setLastMessage(data);
        } catch (e) {
          setLastMessage(event.data);
        }
      };

      ws.current.onerror = (error) => {
        setError('WebSocket connection error');
        setIsConnected(false);
      };

      ws.current.onclose = () => {
        setIsConnected(false);
      };

      return () => {
        if (ws.current && ws.current.readyState === WebSocket.OPEN) {
          ws.current.close();
        }
      };
    } catch (err) {
      setError(err.message);
    }
  }, [url]);

  const send = useCallback((message) => {
    if (ws.current && ws.current.readyState === WebSocket.OPEN) {
      ws.current.send(typeof message === 'string' ? message : JSON.stringify(message));
    }
  }, []);

  return { isConnected, lastMessage, error, send };
};
