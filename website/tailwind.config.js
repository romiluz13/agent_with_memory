/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        neural: {
          50: '#f0f9ff',
          100: '#e0f2fe', 
          200: '#bae6fd',
          300: '#7dd3fc',
          400: '#38bdf8',
          500: '#0ea5e9',
          600: '#0284c7',
          700: '#0369a1',
          800: '#075985',
          900: '#0c4a6e',
          950: '#082f49',
        },
        cyber: {
          50: '#f8fafc',
          100: '#f1f5f9',
          200: '#e2e8f0', 
          300: '#cbd5e1',
          400: '#94a3b8',
          500: '#64748b',
          600: '#475569',
          700: '#334155',
          800: '#1e293b',
          900: '#0f172a',
          950: '#020617',
        },
        neon: {
          blue: '#00d4ff',
          purple: '#b347d9',
          pink: '#ff006e',
          green: '#00ff88',
          yellow: '#ffea00',
        }
      },
      fontFamily: {
        'display': ['Inter', 'system-ui', 'sans-serif'],
        'mono': ['JetBrains Mono', 'SF Mono', 'monospace'],
      },
      fontSize: {
        '8xl': '6rem',
        '9xl': '8rem',
        '10xl': '10rem',
      },
      fontWeight: {
        'black': '900',
        'extrablack': '950',
      },
      animation: {
        'gradient-x': 'gradient-x 15s ease infinite',
        'gradient-y': 'gradient-y 15s ease infinite', 
        'gradient-xy': 'gradient-xy 15s ease infinite',
        'float': 'float 6s ease-in-out infinite',
        'pulse-glow': 'pulse-glow 2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'neural-pulse': 'neural-pulse 3s ease-in-out infinite',
        'data-flow': 'data-flow 4s linear infinite',
        'matrix-rain': 'matrix-rain 10s linear infinite',
        'cyber-glow': 'cyber-glow 2s ease-in-out infinite alternate',
        'neural-network': 'neural-network 8s ease-in-out infinite',
        'brain-wave': 'brain-wave 3s ease-in-out infinite',
        'hologram': 'hologram 4s ease-in-out infinite',
      },
      keyframes: {
        'gradient-y': {
          '0%, 100%': {
            'background-size': '400% 400%',
            'background-position': 'center top'
          },
          '50%': {
            'background-size': '200% 200%', 
            'background-position': 'center center'
          }
        },
        'gradient-x': {
          '0%, 100%': {
            'background-size': '200% 200%',
            'background-position': 'left center'
          },
          '50%': {
            'background-size': '200% 200%',
            'background-position': 'right center'
          }
        },
        'gradient-xy': {
          '0%, 100%': {
            'background-size': '400% 400%',
            'background-position': 'left center'
          },
          '50%': {
            'background-size': '200% 200%',
            'background-position': 'right center'
          }
        },
        'float': {
          '0%, 100%': { 
            'transform': 'translateY(0px) rotate(0deg)',
            'filter': 'hue-rotate(0deg)'
          },
          '33%': { 
            'transform': 'translateY(-20px) rotate(1deg)',
            'filter': 'hue-rotate(90deg)'
          },
          '66%': { 
            'transform': 'translateY(-10px) rotate(-1deg)',
            'filter': 'hue-rotate(180deg)'
          }
        },
        'pulse-glow': {
          '0%, 100%': {
            'opacity': '1',
            'box-shadow': '0 0 0 0 rgba(59, 130, 246, 0.7)',
            'transform': 'scale(1)'
          },
          '70%': {
            'opacity': '0.9',
            'box-shadow': '0 0 0 10px rgba(59, 130, 246, 0)',
            'transform': 'scale(1.05)'
          }
        },
        'neural-pulse': {
          '0%, 100%': {
            'opacity': '0.8',
            'transform': 'scale(1)',
            'filter': 'brightness(1) saturate(1)'
          },
          '50%': {
            'opacity': '1',
            'transform': 'scale(1.1)',
            'filter': 'brightness(1.2) saturate(1.5)'
          }
        },
        'data-flow': {
          '0%': {
            'transform': 'translateX(-100%) scaleX(0)',
            'opacity': '0'
          },
          '50%': {
            'transform': 'translateX(0%) scaleX(1)',
            'opacity': '1'
          },
          '100%': {
            'transform': 'translateX(100%) scaleX(0)',
            'opacity': '0'
          }
        },
        'matrix-rain': {
          '0%': {
            'transform': 'translateY(-100vh)',
            'opacity': '0'
          },
          '10%': {
            'opacity': '1'
          },
          '90%': {
            'opacity': '1'
          },
          '100%': {
            'transform': 'translateY(100vh)',
            'opacity': '0'
          }
        },
        'cyber-glow': {
          '0%': {
            'text-shadow': '0 0 5px rgba(0, 212, 255, 0.8), 0 0 10px rgba(0, 212, 255, 0.6)',
            'filter': 'brightness(1)'
          },
          '100%': {
            'text-shadow': '0 0 10px rgba(0, 212, 255, 1), 0 0 20px rgba(0, 212, 255, 0.8), 0 0 30px rgba(0, 212, 255, 0.6)',
            'filter': 'brightness(1.2)'
          }
        },
        'neural-network': {
          '0%, 100%': {
            'transform': 'rotate(0deg) scale(1)',
            'filter': 'hue-rotate(0deg)'
          },
          '25%': {
            'transform': 'rotate(90deg) scale(1.1)', 
            'filter': 'hue-rotate(90deg)'
          },
          '50%': {
            'transform': 'rotate(180deg) scale(1)',
            'filter': 'hue-rotate(180deg)'
          },
          '75%': {
            'transform': 'rotate(270deg) scale(1.1)',
            'filter': 'hue-rotate(270deg)'
          }
        },
        'brain-wave': {
          '0%, 100%': {
            'transform': 'scaleY(1) scaleX(1)',
            'opacity': '0.8'
          },
          '50%': {
            'transform': 'scaleY(1.2) scaleX(0.9)',
            'opacity': '1'
          }
        },
        'hologram': {
          '0%, 100%': {
            'opacity': '0.8',
            'transform': 'translateZ(0) rotateY(0deg)',
            'filter': 'brightness(1) contrast(1)'
          },
          '25%': {
            'opacity': '1',
            'transform': 'translateZ(10px) rotateY(1deg)',
            'filter': 'brightness(1.1) contrast(1.1)'
          },
          '50%': {
            'opacity': '0.9',
            'transform': 'translateZ(0) rotateY(0deg)',
            'filter': 'brightness(1.2) contrast(1.2)'
          },
          '75%': {
            'opacity': '1',
            'transform': 'translateZ(-10px) rotateY(-1deg)',
            'filter': 'brightness(1.1) contrast(1.1)'
          }
        }
      },
      backdropBlur: {
        'xs': '2px',
        'xl': '24px',
        '2xl': '40px',
        '3xl': '64px',
      },
      boxShadow: {
        'neural': '0 0 20px rgba(59, 130, 246, 0.5), 0 0 40px rgba(59, 130, 246, 0.3), 0 0 80px rgba(59, 130, 246, 0.1)',
        'cyber': '0 0 20px rgba(0, 212, 255, 0.5), 0 0 40px rgba(0, 212, 255, 0.3)',
        'neon': '0 0 5px currentColor, 0 0 10px currentColor, 0 0 15px currentColor, 0 0 20px currentColor',
        'glow-lg': '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05), 0 0 0 1px rgba(255, 255, 255, 0.05)',
      },
      backgroundImage: {
        'neural-gradient': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
        'cyber-gradient': 'linear-gradient(135deg, #00d4ff 0%, #b347d9 50%, #ff006e 100%)',
        'matrix-gradient': 'linear-gradient(180deg, transparent 0%, rgba(0, 255, 136, 0.1) 50%, transparent 100%)',
      }
    },
  },
  plugins: [
    require('@tailwindcss/typography'),
  ],
}