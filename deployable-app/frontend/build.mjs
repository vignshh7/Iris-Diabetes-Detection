import fs from 'node:fs';
import path from 'node:path';

const rootDir = process.cwd();
const distDir = path.join(rootDir, 'dist');
const filesToCopy = ['index.html', 'style.css', 'app.js', 'favicon.svg'];
const apiBaseUrl = process.env.API_BASE_URL || '';

fs.rmSync(distDir, { recursive: true, force: true });
fs.mkdirSync(distDir, { recursive: true });

for (const fileName of filesToCopy) {
  fs.copyFileSync(path.join(rootDir, fileName), path.join(distDir, fileName));
}

fs.writeFileSync(path.join(distDir, 'config.js'), `window.__API_BASE__ = ${JSON.stringify(apiBaseUrl)};\n`, 'utf8');
console.log(`Built frontend with API_BASE_URL=${apiBaseUrl || '(empty)'}`);
