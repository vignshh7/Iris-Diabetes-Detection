import fs from 'node:fs';
import path from 'node:path';

const rootDir = process.cwd();
const distDir = path.join(rootDir, 'dist');
const filesToCopy = ['index.html', 'style.css', 'app.js', 'favicon.svg'];
const apiBaseUrl = process.env.API_BASE_URL || '';

fs.rmSync(distDir, { recursive: true, force: true });
fs.mkdirSync(distDir, { recursive: true });

for (const fileName of filesToCopy) {
  const sourcePath = path.join(rootDir, fileName);
  const targetPath = path.join(distDir, fileName);

  if (!fs.existsSync(sourcePath)) {
    console.warn(`Skipping missing file: ${fileName}`);
    continue;
  }

  fs.copyFileSync(sourcePath, targetPath);
}

fs.writeFileSync(path.join(distDir, 'config.js'), `window.__API_BASE__ = ${JSON.stringify(apiBaseUrl)};\n`, 'utf8');
console.log(`Built frontend with API_BASE_URL=${apiBaseUrl || '(empty)'}`);
