import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { Play, Square, Video, BarChart3, List, Activity, Cpu } from 'lucide-react';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const API_BASE = window.location.origin.replace('5173', '8000');

function App() {
  const [videos, setVideos] = useState([]);
  const [stats, setStats] = useState({ totals: {}, class_names: {} });
  const [activeVideo, setActiveVideo] = useState(null);
  const [isRunning, setIsRunning] = useState(false);
  const [chartData, setChartData] = useState([]);
  const statsInterval = useRef(null);

  useEffect(() => {
    fetchVideos();
    return () => clearInterval(statsInterval.current);
  }, []);

  const fetchVideos = async () => {
    try {
      const res = await axios.get(`${API_BASE}/api/videos`);
      setVideos(res.data.videos);
    } catch (err) {
      console.error("Failed to fetch videos", err);
    }
  };

  const startStream = async (videoName) => {
    try {
      await axios.get(`${API_BASE}/api/start?video_name=${videoName}`);
      setActiveVideo(videoName);
      setIsRunning(true);
      startPolling();
    } catch (err) {
      console.error("Start failed", err);
    }
  };

  const stopStream = async () => {
    try {
      await axios.post(`${API_BASE}/api/stop`);
      setIsRunning(false);
      clearInterval(statsInterval.current);
    } catch (err) {
      console.error("Stop failed", err);
    }
  };

  const startPolling = () => {
    if (statsInterval.current) clearInterval(statsInterval.current);
    statsInterval.current = setInterval(async () => {
      try {
        const res = await axios.get(`${API_BASE}/api/stats`);
        setStats(res.data);
        
        const total = Object.values(res.data.totals).reduce((a, b) => a + b, 0);
        setChartData(prev => [...prev.slice(-19), { time: new Date().toLocaleTimeString(), count: total }]);
      } catch (err) {
        console.error("Polling failed", err);
      }
    }, 1000);
  };

  return (
    <div className="dashboard">
      <aside className="sidebar">
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '10px' }}>
          <div style={{ background: 'var(--primary)', padding: '8px', borderRadius: '10px' }}>
            <Activity size={24} color="white" />
          </div>
          <div>
            <h2 style={{ fontSize: '20px', fontWeight: '700' }}>SegmentAI</h2>
            <p style={{ fontSize: '12px', color: 'var(--text-muted)' }}>Industrial Monitoring</p>
          </div>
        </div>

        <div style={{ marginTop: '20px' }}>
          <h3 style={{ fontSize: '14px', textTransform: 'uppercase', color: 'var(--text-muted)', marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Video size={16} /> Источники видео
          </h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
            {videos.map(v => (
              <div 
                key={v} 
                className={`video-list-item ${activeVideo === v ? 'active-video' : ''}`}
                onClick={() => !isRunning && startStream(v)}
              >
                {v}
              </div>
            ))}
          </div>
        </div>

        <div style={{ marginTop: 'auto' }}>
          <div style={{ background: 'var(--glass)', padding: '16px', borderRadius: '16px', border: '1px solid var(--glass-border)' }}>
             <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
                <Cpu size={16} color="var(--primary)" />
                <span style={{ fontSize: '14px', fontWeight: '600' }}>Статус системы</span>
             </div>
             <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
                GPU: CUDA Compatible<br/>
                Model: YOLO Segmentation
             </div>
          </div>
        </div>
      </aside>

      <main className="main-content">
        <header className="header">
          <div>
            <h1 style={{ fontSize: '28px' }}>Панель управления</h1>
            <p style={{ color: 'var(--text-muted)' }}>Мониторинг объектов в реальном времени</p>
          </div>
          <div style={{ display: 'flex', gap: '12px' }}>
            {!isRunning ? (
              <>
                <button 
                  className="btn btn-primary" 
                  onClick={() => activeVideo && startStream(activeVideo)}
                  disabled={!activeVideo}
                >
                  <Play size={20} fill="white" /> Видеофайл
                </button>
                <button 
                  className="btn btn-primary" 
                  style={{ background: 'linear-gradient(135deg, #10b981, #059669)' }}
                  onClick={async () => {
                    try {
                      await axios.get(`${API_BASE}/api/camera/start`);
                      setIsRunning(true);
                      startPolling();
                    } catch (e) { alert("Ошибка подключения камеры"); }
                  }}
                >
                  <Activity size={20} /> Живая камера
                </button>
              </>
            ) : (
              <button className="btn btn-primary" style={{ background: '#ef4444' }} onClick={stopStream}>
                <Square size={20} fill="white" /> Остановить
              </button>
            )}
          </div>
        </header>

        <section style={{ display: 'grid', gridTemplateColumns: '1fr 340px', gap: '20px' }}>
          <div className="video-container">
            {isRunning ? (
              <img 
                src={`${API_BASE}/video_feed?t=${new Date().getTime()}`} 
                alt="Stream" 
                className="video-feed" 
              />
            ) : (
              <div style={{ width: '100%', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)', background: '#000' }}>
                Поток остановлен. Выберите видео и нажмите "Запустить".
              </div>
            )}
          </div>

          <div className="stats-panel" style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
             <div className="stat-card">
                <div style={{ color: 'var(--text-muted)', fontSize: '14px', fontWeight: '600', marginBottom: '8px' }}>Всего объектов</div>
                <div className="stat-value">{Object.values(stats.totals).reduce((a, b) => a + b, 0)}</div>
                <div style={{ fontSize: '12px', color: '#10b981' }}>+12% за час</div>
             </div>

             <div style={{ background: 'var(--bg-card)', padding: '20px', borderRadius: '24px', flexGrow: 1, border: '1px solid var(--glass-border)' }}>
                <h3 style={{ fontSize: '16px', marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <BarChart3 size={18} /> Классы
                </h3>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                  {Object.entries(stats.totals).map(([id, count]) => (
                    <div key={id} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <span style={{ fontSize: '14px' }}>{stats.class_names[id] || `Class ${id}`}</span>
                      <span style={{ fontWeight: '700', color: 'var(--primary)' }}>{count}</span>
                    </div>
                  ))}
                </div>
             </div>
          </div>
        </section>

        <section style={{ background: 'var(--bg-card)', padding: '24px', borderRadius: '24px', border: '1px solid var(--glass-border)', height: '240px' }}>
          <h3 style={{ fontSize: '16px', marginBottom: '20px', display: 'flex', alignItems: 'center', gap: '8px' }}>
             <Activity size={18} /> Динамика обнаружения
          </h3>
          <div style={{ width: '100%', height: '140px' }}>
             <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={chartData}>
                  <defs>
                    <linearGradient id="colorCount" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#38bdf8" stopOpacity={0.3}/>
                      <stop offset="95%" stopColor="#38bdf8" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                  <XAxis dataKey="time" hide />
                  <YAxis hide />
                  <Tooltip 
                    contentStyle={{ background: '#1e293b', border: 'none', borderRadius: '12px', color: '#fff' }}
                    itemStyle={{ color: '#38bdf8' }}
                  />
                  <Area type="monotone" dataKey="count" stroke="#38bdf8" fillOpacity={1} fill="url(#colorCount)" strokeWidth={3} />
                </AreaChart>
             </ResponsiveContainer>
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;
