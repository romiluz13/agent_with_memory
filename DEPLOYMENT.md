# 🚀 Deployment Guide

## Deploy Landing Page on Replit

### Option 1: Direct Deploy (Recommended)

1. **Go to Replit**: https://replit.com
2. **Create New Repl**: 
   - Click "Create Repl"
   - Select "Import from GitHub"
   - Enter: `https://github.com/romiluz13/agent_with_memory`
3. **Navigate to Website**: 
   ```bash
   cd website
   ```
4. **Install & Run**:
   ```bash
   npm install
   npm run dev
   ```
5. **Your site is live!** Replit will provide a public URL

### Option 2: Fork & Deploy

1. **Fork the Repository** on GitHub
2. **Import to Replit** from your fork
3. **Follow steps 3-5** above

### 🔧 Configuration

The website includes:
- `.replit` - Replit configuration
- `replit.nix` - Environment setup
- `package.json` - Dependencies and scripts

### 🌐 Custom Domain (Optional)

1. In Replit, go to your project settings
2. Under "Domains", add your custom domain
3. Update DNS settings as instructed

### 🔄 Auto-Deploy

Replit automatically redeploys when you:
- Push changes to GitHub
- Edit files directly in Replit

### 📊 Monitoring

- **Uptime**: Replit provides built-in uptime monitoring
- **Analytics**: Add Google Analytics to track visitors
- **Performance**: Monitor via browser dev tools

## Alternative Deployment Options

### Vercel (Next.js Optimized)
```bash
npm install -g vercel
cd website
vercel
```

### Netlify
```bash
cd website
npm run build
# Upload dist/ folder to Netlify
```

### GitHub Pages (Static)
```bash
cd website
npm run build
npm run export
# Deploy out/ folder to GitHub Pages
```

## Environment Variables

For production deployment, you may want to set:
- `NODE_ENV=production`
- `NEXT_PUBLIC_GA_ID` (for Google Analytics)
- `NEXT_PUBLIC_DOMAIN` (for SEO)

## Troubleshooting

**Port Issues**: Replit uses port 3000 by default (configured in package.json)
**Build Errors**: Check Node.js version (should be 18+)
**Slow Loading**: Enable Next.js optimization in production
