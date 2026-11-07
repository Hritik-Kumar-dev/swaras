const searchTerm = encodeURIComponent('Rang Re Anaya Rao');

function ArrowIcon() {
  return <svg aria-hidden="true" viewBox="0 0 18 18" fill="none"><path d="M3.75 9h10.5M9 3.75 14.25 9 9 14.25" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" /></svg>;
}

function BrandMark() {
  return <a className="brand" href="#top" aria-label="Swaras Sessions home"><span className="brand-mark" aria-hidden="true"><span /></span><span className="brand-word">swaras<span>sessions</span></span></a>;
}

function AlbumArtwork() {
  return (
    <div className="artwork-wrap" aria-label="Abstract artwork for Rang Re by Anaya Rao">
      <div className="artwork">
        <div className="art-sun" /><div className="art-sweep art-sweep-one" /><div className="art-sweep art-sweep-two" />
        <div className="art-ripple ripple-one" /><div className="art-ripple ripple-two" /><div className="art-ripple ripple-three" />
        <span className="art-side-note">SWARAS SESSIONS · VOL. 04</span>
        <div className="art-title"><span>ANAYA RAO</span><strong>rang<br /><i>re.</i></strong><span>OUT IN THE OPEN</span></div>
        <div className="art-edition">01 <span>—</span> 04</div>
      </div>
      <span className="art-caption"><span className="caption-dot" /> COVER ART · 2026</span>
    </div>
  );
}

function App() {
  return (
    <main id="top">
      <nav className="topbar" aria-label="Main navigation">
        <BrandMark />
        <a className="nav-link" href="#story">THE STORY <span>↘</span></a>
        <a className="nav-listen" href={`https://open.spotify.com/search/${searchTerm}`} target="_blank" rel="noreferrer">LISTEN NOW <ArrowIcon /></a>
      </nav>

      <section className="hero" aria-labelledby="hero-title">
        <div className="hero-copy">
          <div className="eyebrow"><span className="eyebrow-line" /> A SWARAS SESSIONS RELEASE <span className="eyebrow-year">NO. 004 / 2026</span></div>
          <h1 id="hero-title">A little more<br />feeling. A little<br /><em>less noise.</em></h1>
          <p className="intro">Some songs don't ask for your attention.<br className="desktop-break" /> They just find their way in.</p>
          <div className="release-byline"><span className="byline-rule" /><span><b>ANAYA RAO</b><small>NEW SINGLE · RANG RE</small></span><span className="byline-date">OUT NOW</span></div>
          <div className="hero-actions">
            <a className="button button-dark" href={`https://open.spotify.com/search/${searchTerm}`} target="_blank" rel="noreferrer">LISTEN ON SPOTIFY <ArrowIcon /></a>
            <a className="text-link" href={`https://music.apple.com/us/search?term=${searchTerm}`} target="_blank" rel="noreferrer">Apple Music <span>↗</span></a>
          </div>
          <p className="micro-note">HEADPHONES ON. WORLD OFF.</p>
        </div>

        <div className="hero-art"><AlbumArtwork /></div>
        <div className="hero-bottom" aria-hidden="true"><span>INDEPENDENT SOUND, MADE WITH INTENTION</span><span className="bottom-coordinates">19°04′ N&nbsp; 72°52′ E</span></div>
      </section>

      <section className="ticker" aria-label="Swaras Sessions motto"><div className="ticker-track"><span>LISTEN CLOSE</span><i>✳</i><span>FEEL DEEP</span><i>✳</i><span>RANG RE</span><i>✳</i><span>LISTEN CLOSE</span><i>✳</i><span>FEEL DEEP</span><i>✳</i><span>RANG RE</span><i>✳</i></div></section>

      <section className="story" id="story" aria-labelledby="story-title">
        <div className="story-label"><span className="eyebrow-line" /> A NOTE FROM THE STUDIO</div>
        <div className="story-content">
          <h2 id="story-title">Made for the<br /><em>in-between.</em></h2>
          <div className="story-aside"><p>That last train home. The long way back. The quiet after a good evening. <i>Rang Re</i> is a small reminder to stay for the feeling.</p><a className="story-link" href={`https://open.spotify.com/search/${searchTerm}`} target="_blank" rel="noreferrer">FIND YOUR NEW FAVOURITE <ArrowIcon /></a></div>
        </div>
        <footer className="footer"><BrandMark /><span>INDEPENDENT MUSIC. IN ITS OWN TIME.</span><span>© SWARAS SESSIONS 2026</span></footer>
      </section>
    </main>
  );
}

export default App;
