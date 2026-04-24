const fs = require('fs');
const path = require('path');

// Simple .env parser for local testing
try {
  const envContent = fs.readFileSync(path.join(__dirname, '../.env'), 'utf8');
  envContent.split('\n').forEach(line => {
    const match = line.match(/^([^=]+)=(.*)$/);
    if (match) {
      process.env[match[1].trim()] = match[2].trim();
    }
  });
} catch (e) {
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
