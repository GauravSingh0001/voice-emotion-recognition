/**
 * inject-env.js — Build-time environment variable injector
 *
 * Replaces these three placeholders in index.html and result.html with
 * values from .env or system environment variables:
 *   __BACKEND_URL__       → BACKEND_URL       (default: http://localhost:8000)
 *   __SUPABASE_URL__      → SUPABASE_URL      (required)
 *   __SUPABASE_ANON_KEY__ → SUPABASE_ANON_KEY (required)
 *
 * Run before deploying: node scripts/inject-env.js
 */
const fs = require('fs');
const path = require('path');

// Simple .env parser for local testing
try {
  const envContent = fs.readFileSync(path.join(__dirname, '../.env'), 'utf8');
  envContent.split(/\r?\n/).forEach(line => {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith('#')) return;
    const parts = trimmed.split('=');
    if (parts.length >= 2) {
      const key = parts[0].trim();
      const val = parts.slice(1).join('=').trim();
      process.env[key] = val;
    }
  });
}
 catch (e) {
  console.log('No .env file found, relying on system environment variables.');
}

const filesToInject = ['index.html', 'result.html'];

filesToInject.forEach(filename => {
  const filepath = path.join(__dirname, '../', filename);
  try {
    let content = fs.readFileSync(filepath, 'utf8');
    
    // Replace placeholders
    content = content
      .replace(/__BACKEND_URL__/g, process.env.BACKEND_URL || 'http://localhost:8000')
      .replace(/__SUPABASE_URL__/g, process.env.SUPABASE_URL || '')
      .replace(/__SUPABASE_ANON_KEY__/g, process.env.SUPABASE_ANON_KEY || '');
      
    fs.writeFileSync(filepath, content);
    console.log(`Injected environment variables into ${filename}`);
  } catch (err) {
    console.error(`Error processing ${filename}:`, err);
  }
});
