import { useState } from "react";
import {
  LayoutDashboard, BookOpen, BrainCircuit, ChartNoAxesCombined, Settings, Bell,
  Search, Flame, ArrowUpRight, Clock3, CircleCheck, Play, ChevronRight, Menu, X,
} from "lucide-react";
import "./App.css";

const initialQuizzes = [
  { title: "Linear Algebra", category: "Mathematics", score: 85, color: "pink" },
  { title: "Data Structures", category: "Computer Science", score: 72, color: "sky" },
  { title: "Probability Basics", category: "Statistics", score: 91, color: "mint" },
];

const topics = [
  { title: "Linear Algebra", subtitle: "Vectors, matrices, and the maths behind them", icon: "∑", level: "Medium", color: "pink" },
  { title: "Data Structures", subtitle: "Arrays, trees, and when to use which", icon: "⌘", level: "Hard", color: "sky" },
  { title: "Probability", subtitle: "Dice, coins, and how likely things are", icon: "π", level: "Easy", color: "mint" },
];

const navItems = [
  { name: "Overview", icon: LayoutDashboard },
  { name: "My Quizzes", icon: BookOpen },
  { name: "Recommendations", icon: BrainCircuit },
  { name: "Model Health", icon: ChartNoAxesCombined },
];

export default function App() {
  const [activePage, setActivePage] = useState("Overview");
  const [mobileMenu, setMobileMenu] = useState(false);
  const [quizzes] = useState(initialQuizzes);
  const [notice, setNotice] = useState("");

  function startQuiz(topic) {
    setNotice(`"${topic}" quiz picked. Questions will connect to the Python quiz engine next.`);
    setActivePage("My Quizzes");
  }

  const avg = Math.round(quizzes.reduce((s, q) => s + q.score, 0) / quizzes.length);

  function renderContent() {
    if (activePage === "Model Health") {
      return (
        <>
          <div className="page-heading"><h2>Model health</h2><p className="muted">Is the recommendation pipeline behaving?</p></div>
          <div className="stats-grid">
            <Stat icon={BrainCircuit} label="Active model" value="v1" detail="Demo value" color="yellow" />
            <Stat icon={ChartNoAxesCombined} label="Drift status" value="Not linked" detail="Waiting on the pipeline" color="sky" />
            <Stat icon={CircleCheck} label="Retraining" value="Not linked" detail="Waiting on the backend" color="mint" />
          </div>
          <section className="panel">
            <h3>Pipeline monitoring</h3>
            <p className="muted">Real drift metrics, challenger results, and promotion status show up here once the Python backend is connected.</p>
            <div className="status-row"><span className="status-dot" /> Backend not connected yet</div>
          </section>
        </>
      );
    }

    if (activePage === "Recommendations" || activePage === "My Quizzes") {
      const isRec = activePage === "Recommendations";
      return (
        <>
          <div className="page-heading">
            <h2>{isRec ? "Try these next" : "My quizzes"}</h2>
            <p className="muted">{isRec ? "Picked from what you've been practising." : "Pick a topic and start a session."}</p>
          </div>
          <div className="topic-grid">
            {topics.map((t) => (
              <article className={`topic-card ${t.color}`} key={t.title}>
                <div className="topic-symbol">{t.icon}</div>
                <span className="topic-level">{t.level}</span>
                <h3>{t.title}</h3>
                <p className="muted">{t.subtitle}</p>
                <button className="btn dark" onClick={() => startQuiz(t.title)}><Play size={15} /> {isRec ? "Practise this" : "Start quiz"}</button>
              </article>
            ))}
          </div>
        </>
      );
    }

    return (
      <>
        <div className="hero">
          <div className="hero-copy">
            <h2>Small steps,<br />big brain.</h2>
            <p>One quiz at a time. Pick up where you left off.</p>
            <button className="btn" onClick={() => setActivePage("My Quizzes")}>Open quizzes <ArrowUpRight size={16} /></button>
          </div>
          <div className="hero-art" aria-hidden="true">
            <span className="sticker s-sum">∑</span>
            <span className="sticker s-book"><BookOpen size={56} strokeWidth={2} /></span>
            <span className="sticker s-pi">π</span>
            <span className="sticker s-star">✦</span>
          </div>
        </div>

        <div className="section-heading"><h3>How you're doing</h3></div>
        <div className="stats-grid">
          <Stat icon={BookOpen} label="Quizzes done" value={quizzes.length + 9} detail="Across all topics" color="yellow" />
          <Stat icon={ChartNoAxesCombined} label="Average score" value={`${avg}%`} detail="From sample history" color="pink" />
          <Stat icon={Flame} label="Streak" value="4 days" detail="Demo value" color="mint" />
        </div>

        <div className="content-grid">
          <section className="panel">
            <div className="panel-heading">
              <div><h3>Recent quizzes</h3><p className="muted">Your last few sessions.</p></div>
              <button className="link-btn" onClick={() => setActivePage("My Quizzes")}>See all <ChevronRight size={15} /></button>
            </div>
            {quizzes.map((q) => (
              <div className="quiz-row" key={q.title}>
                <div className={`quiz-icon ${q.color}`}><BookOpen size={20} /></div>
                <div className="quiz-info"><strong>{q.title}</strong><span>{q.category}</span></div>
                <div className="score-wrap">
                  <strong>{q.score}%</strong>
                  <div className="score-track"><div className={`score-fill ${q.color}`} style={{ width: `${q.score}%` }} /></div>
                </div>
              </div>
            ))}
          </section>

          <section className="panel">
            <div className="panel-heading"><div><h3>Up next</h3><p className="muted">Keep the streak going.</p></div></div>
            {topics.slice(0, 2).map((t) => (
              <div className="next-topic" key={t.title}>
                <div className={`topic-symbol small ${t.color}`}>{t.icon}</div>
                <div className="next-info"><strong>{t.title}</strong><span><Clock3 size={14} /> 10 questions, {t.level}</span></div>
                <button className="round-button" aria-label={`Practise ${t.title}`} onClick={() => startQuiz(t.title)}><ArrowUpRight size={18} /></button>
              </div>
            ))}
            <button className="btn full" onClick={() => setActivePage("Recommendations")}>See all picks</button>
          </section>
        </div>
      </>
    );
  }

  return (
    <div className="app-shell">
      {mobileMenu && <button className="mobile-scrim" aria-label="Close menu" onClick={() => setMobileMenu(false)} />}
      <aside className={`sidebar ${mobileMenu ? "sidebar-open" : ""}`}>
        <div className="brand">
          <div className="brand-mark"><BrainCircuit size={24} /></div>
          <strong>Mindstep</strong>
          <button className="close-menu" aria-label="Close menu" onClick={() => setMobileMenu(false)}><X size={20} /></button>
        </div>
        <nav>
          {navItems.map(({ name, icon: Icon }) => (
            <button key={name} className={`nav-link ${activePage === name ? "active" : ""}`} onClick={() => { setActivePage(name); setMobileMenu(false); setNotice(""); }}>
              <Icon size={20} strokeWidth={2.2} /><span>{name}</span>{name === "Model Health" && <span className="nav-dot" />}
            </button>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <div className="sidebar-note"><strong>It learns you back</strong><p>The more you practise, the better your picks get.</p></div>
          <button className="nav-link" onClick={() => setNotice("Settings are coming in a later version.")}><Settings size={20} /><span>Settings</span></button>
          <div className="profile"><div className="avatar">K</div><div><strong>Learner</strong><span>Student account</span></div></div>
        </div>
      </aside>

      <main className="main-area">
        <header className="topbar">
          <button className="icon-btn menu-toggle" aria-label="Open menu" onClick={() => setMobileMenu(true)}><Menu size={20} /></button>
          <div className="breadcrumb"><span className="crumb-root">Workspace</span><ChevronRight size={15} /><strong>{activePage}</strong></div>
          <div className="topbar-actions">
            <div className="search-box">
              <Search size={17} />
              <input placeholder="Search topics" aria-label="Search topics" onKeyDown={(e) => { if (e.key === "Enter") { setActivePage("Recommendations"); setNotice(`Showing topics related to "${e.currentTarget.value}".`); } }} />
            </div>
            <button className="icon-btn" aria-label="Notifications" onClick={() => setNotice("Nothing new. You're all caught up!")}><Bell size={19} /><span className="ping" /></button>
          </div>
        </header>

        <div className="page-content">
          {notice && <div className="notice" role="status">{notice}<button onClick={() => setNotice("")} aria-label="Dismiss notice"><X size={16} /></button></div>}
          {renderContent()}
          <footer className="footer"><span>Mindstep, adaptive learning</span><span>Go at your own pace ✦</span></footer>
        </div>
      </main>
    </div>
  );
}

function Stat({ icon: Icon, label, value, detail, color }) {
  return (
    <article className={`stat-card ${color}`}>
      <div className="stat-icon"><Icon size={20} /></div>
      <div className="stat-label">{label}</div>
      <div className="stat-value">{value}</div>
      <div className="stat-detail">{detail}</div>
    </article>
  );
}
