import React, { useState, useRef, useEffect } from 'react';
import Head from 'next/head';
import { motion, useScroll, useTransform } from 'framer-motion';
import { 
  Brain, 
  Zap, 
  Database, 
  Cpu, 
  Shield, 
  Rocket,
  Code2,
  Layers,
  Clock,
  Star,
  Github,
  ExternalLink,
  ChevronRight,
  CheckCircle,
  Workflow,
  Activity,
  MemoryStick,
  Gauge
} from 'lucide-react';

// Neural Network Canvas Component
const NeuralNetwork = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number>();
  
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    
    // Neural network nodes
    const nodes: Array<{x: number, y: number, vx: number, vy: number, radius: number, color: string}> = [];
    const connections: Array<{from: number, to: number, strength: number}> = [];
    
    // Create nodes
    for (let i = 0; i < 50; i++) {
      nodes.push({
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        vx: (Math.random() - 0.5) * 0.5,
        vy: (Math.random() - 0.5) * 0.5,
        radius: Math.random() * 3 + 2,
        color: `hsla(${200 + Math.random() * 60}, 70%, 60%, 0.8)`
      });
    }
    
    // Create connections
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const distance = Math.sqrt(
          Math.pow(nodes[i].x - nodes[j].x, 2) + 
          Math.pow(nodes[i].y - nodes[j].y, 2)
        );
        if (distance < 150) {
          connections.push({
            from: i,
            to: j,
            strength: 1 - distance / 150
          });
        }
      }
    }
    
    const animate = () => {
      ctx.fillStyle = 'rgba(15, 23, 42, 0.05)';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      
      // Update and draw connections
      connections.length = 0;
      for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
          const distance = Math.sqrt(
            Math.pow(nodes[i].x - nodes[j].x, 2) + 
            Math.pow(nodes[i].y - nodes[j].y, 2)
          );
          if (distance < 150) {
            const strength = 1 - distance / 150;
            ctx.beginPath();
            ctx.moveTo(nodes[i].x, nodes[i].y);
            ctx.lineTo(nodes[j].x, nodes[j].y);
            ctx.strokeStyle = `rgba(99, 102, 241, ${strength * 0.3})`;
            ctx.lineWidth = strength * 2;
            ctx.stroke();
          }
        }
      }
      
      // Update and draw nodes
      nodes.forEach(node => {
        // Update position
        node.x += node.vx;
        node.y += node.vy;
        
        // Bounce off walls
        if (node.x < 0 || node.x > canvas.width) node.vx *= -1;
        if (node.y < 0 || node.y > canvas.height) node.vy *= -1;
        
        // Draw node
        ctx.beginPath();
        ctx.arc(node.x, node.y, node.radius, 0, Math.PI * 2);
        ctx.fillStyle = node.color;
        ctx.fill();
        
        // Glow effect
        const gradient = ctx.createRadialGradient(node.x, node.y, 0, node.x, node.y, node.radius * 3);
        gradient.addColorStop(0, 'rgba(99, 102, 241, 0.3)');
        gradient.addColorStop(1, 'rgba(99, 102, 241, 0)');
        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.arc(node.x, node.y, node.radius * 3, 0, Math.PI * 2);
        ctx.fill();
      });
      
      animationRef.current = requestAnimationFrame(animate);
    };
    
    animate();
    
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, []);
  
  return (
    <canvas
      ref={canvasRef}
      className="absolute inset-0 opacity-30"
      style={{ background: 'transparent' }}
    />
  );
};

const Hero = () => {
  const { scrollY } = useScroll();
  const y = useTransform(scrollY, [0, 1000], [0, -100]);
  
  return (
    <section className="relative min-h-screen flex items-center justify-center overflow-hidden">
      {/* Neural Network Background */}
      <div className="absolute inset-0 bg-gradient-to-br from-slate-950 via-slate-900 to-slate-800">
        <NeuralNetwork />
      </div>
      
      {/* Content */}
      <motion.div 
        style={{ y }}
        className="relative z-10 text-center px-6 max-w-7xl mx-auto"
      >
        {/* Tech Stack Badges */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="mb-12"
        >
          <div className="flex flex-wrap justify-center gap-4 mb-8">
            {/* LangGraph Badge */}
            <div className="flex items-center gap-2 px-4 py-2 bg-white/10 backdrop-blur-sm rounded-lg border border-white/20">
              <div className="w-6 h-6 bg-gradient-to-br from-blue-500 to-blue-600 rounded flex items-center justify-center">
                <Workflow className="w-4 h-4 text-white" />
              </div>
              <span className="text-white font-medium">LangGraph</span>
            </div>
            
            {/* MongoDB Badge */}
            <div className="flex items-center gap-2 px-4 py-2 bg-white/10 backdrop-blur-sm rounded-lg border border-white/20">
              <div className="w-6 h-6 bg-gradient-to-br from-green-500 to-green-600 rounded flex items-center justify-center">
                <Database className="w-4 h-4 text-white" />
              </div>
              <span className="text-white font-medium">MongoDB Atlas</span>
            </div>
            
            {/* Voyage AI Badge */}
            <div className="flex items-center gap-2 px-4 py-2 bg-white/10 backdrop-blur-sm rounded-lg border border-white/20">
              <div className="w-6 h-6 bg-gradient-to-br from-purple-500 to-purple-600 rounded flex items-center justify-center">
                <Zap className="w-4 h-4 text-white" />
              </div>
              <span className="text-white font-medium">Voyage AI</span>
            </div>
          </div>
          
          <h1 className="text-5xl md:text-7xl font-black mb-8 leading-tight">
            <span className="block text-white">AI Agent Boilerplate</span>
            <span className="block bg-gradient-to-r from-blue-400 via-purple-500 to-pink-500 bg-clip-text text-transparent animate-gradient">
              with Memory
            </span>
          </h1>
          
          <p className="text-xl md:text-2xl text-slate-300 mb-12 max-w-4xl mx-auto leading-relaxed">
            Production-ready agent framework with <span className="text-blue-400 font-semibold">5-component memory system</span>, 
            built on <span className="text-green-400 font-semibold">MongoDB Atlas</span> and <span className="text-blue-400 font-semibold">LangGraph</span>
          </p>
        </motion.div>

        {/* Action Buttons */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.3 }}
          className="flex flex-col sm:flex-row gap-6 justify-center mb-16"
        >
          <a
            href="https://github.com/romiluz13/agent_with_memory"
            className="group px-8 py-4 bg-gradient-to-r from-blue-600 to-purple-600 text-white font-semibold rounded-xl transition-all duration-300 flex items-center gap-3 hover:shadow-lg hover:shadow-blue-500/25 hover:scale-105"
          >
            <Github className="w-5 h-5" />
            Clone Repository
            <ChevronRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
          </a>
          
          <a
            href="#architecture"
            className="px-8 py-4 bg-white/10 backdrop-blur-sm text-white font-semibold rounded-xl border border-white/20 hover:bg-white/20 transition-all duration-300 flex items-center gap-3"
          >
            <Code2 className="w-5 h-5" />
            View Architecture
          </a>
        </motion.div>

        {/* Real Stats with Glassmorphism */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.8, delay: 0.6 }}
          className="grid grid-cols-2 md:grid-cols-4 gap-8 max-w-3xl mx-auto"
        >
          {[
            { label: "Memory Components", value: "5", icon: <Brain className="w-5 h-5" /> },
            { label: "Setup Time", value: "5min", icon: <Clock className="w-5 h-5" /> },
            { label: "Vector Dimensions", value: "1024", icon: <Database className="w-5 h-5" /> },
            { label: "Production Ready", value: "✓", icon: <Shield className="w-5 h-5" /> }
          ].map((stat, i) => (
            <motion.div 
              key={i} 
              className="text-center bg-white/5 backdrop-blur-sm rounded-xl p-4 border border-white/10"
              whileHover={{ scale: 1.05 }}
              transition={{ type: "spring", stiffness: 300 }}
            >
              <div className="flex justify-center mb-2 text-blue-400">
                {stat.icon}
              </div>
              <div className="text-2xl font-bold text-white mb-1">{stat.value}</div>
              <div className="text-sm text-slate-400">{stat.label}</div>
            </motion.div>
          ))}
        </motion.div>
      </motion.div>
    </section>
  );
};

// Real Memory Architecture with Neural Design
const MemoryArchitecture = () => {
  const [activeMemory, setActiveMemory] = useState<string | null>(null);
  
  const memoryTypes = [
    {
      id: 'episodic',
      name: "Episodic Memory",
      description: "Conversation history, user interactions, temporal sequences",
      implementation: "ConversationMemoryUnit",
      storage: "MongoDB conversations collection",
      icon: <Clock className="w-6 h-6" />,
      color: "from-blue-500 to-blue-600"
    },
    {
      id: 'semantic', 
      name: "Semantic Memory",
      description: "Domain knowledge, facts, concepts, relationships",
      implementation: "KnowledgeBase + Vector Search",
      storage: "MongoDB Atlas Vector Search",
      icon: <Database className="w-6 h-6" />,
      color: "from-green-500 to-green-600"
    },
    {
      id: 'procedural',
      name: "Procedural Memory", 
      description: "Learned workflows, step-by-step processes, patterns",
      implementation: "Workflow execution patterns",
      storage: "Process definitions & execution logs",
      icon: <Workflow className="w-6 h-6" />,
      color: "from-purple-500 to-purple-600"
    },
    {
      id: 'working',
      name: "Working Memory",
      description: "Active context, current session state, temporary data",
      implementation: "LangGraph state management",
      storage: "In-memory + MongoDB checkpoints",
      icon: <Cpu className="w-6 h-6" />,
      color: "from-orange-500 to-orange-600"
    },
    {
      id: 'cache',
      name: "Semantic Cache",
      description: "Query results, computed responses, performance optimization",
      implementation: "Vector similarity caching",
      storage: "Redis-compatible caching layer",
      icon: <Zap className="w-6 h-6" />,
      color: "from-red-500 to-red-600"
    }
  ];
  
  return (
    <section id="architecture" className="py-24 bg-gradient-to-b from-slate-900 to-slate-950 relative">
      {/* Subtle grid background */}
      <div className="absolute inset-0 bg-grid-white/[0.02] bg-[size:50px_50px]" />
      
      <div className="max-w-7xl mx-auto px-6 relative">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true }}
          className="text-center mb-16"
        >
          <h2 className="text-4xl md:text-5xl font-black text-white mb-6">
            Memory Architecture
          </h2>
          <p className="text-xl text-slate-400 max-w-3xl mx-auto">
            5-component memory system based on cognitive science research, implemented with production-grade infrastructure
          </p>
        </motion.div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {memoryTypes.map((memory, index) => (
            <motion.div
              key={memory.id}
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: index * 0.1 }}
              viewport={{ once: true }}
              className="group cursor-pointer"
              onMouseEnter={() => setActiveMemory(memory.id)}
              onMouseLeave={() => setActiveMemory(null)}
            >
              <div className="bg-slate-800/50 backdrop-blur-sm p-8 rounded-2xl border border-slate-700 hover:border-blue-500/50 transition-all duration-300 h-full relative overflow-hidden">
                {/* Gradient overlay on hover */}
                <div className={`absolute inset-0 bg-gradient-to-br ${memory.color} opacity-0 group-hover:opacity-10 transition-opacity duration-300`} />
                
                <div className={`inline-flex p-3 rounded-xl bg-gradient-to-r ${memory.color} text-white mb-4 relative`}>
                  {memory.icon}
                </div>
                <h3 className="text-xl font-bold text-white mb-3">{memory.name}</h3>
                <p className="text-slate-400 mb-4">{memory.description}</p>
                
                <div className="space-y-2 text-sm">
                  <div className="flex items-center gap-2">
                    <Code2 className="w-4 h-4 text-slate-500" />
                    <span className="text-slate-500">Implementation:</span>
                  </div>
                  <code className="block text-blue-400 bg-slate-900/50 px-2 py-1 rounded font-mono text-xs">
                    {memory.implementation}
                  </code>
                  
                  <div className="flex items-center gap-2 mt-3">
                    <Database className="w-4 h-4 text-slate-500" />
                    <span className="text-slate-500">Storage:</span>
                  </div>
                  <div className="text-slate-400 text-xs">{memory.storage}</div>
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
};

// Real Code Implementation with Terminal Design
const CodeImplementation = () => {
  return (
    <section className="py-24 bg-slate-950 relative">
      {/* Animated gradient background */}
      <div className="absolute inset-0">
        <div className="absolute inset-0 bg-gradient-to-r from-blue-600/10 via-purple-600/10 to-pink-600/10 animate-gradient-x" />
      </div>
      
      <div className="max-w-6xl mx-auto px-6 relative">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true }}
          className="text-center mb-16"
        >
          <h2 className="text-4xl md:text-5xl font-black text-white mb-6">
            Implementation
          </h2>
          <p className="text-xl text-slate-400 max-w-3xl mx-auto">
            Real code that actually works. No marketing fluff.
          </p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          whileInView={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.8, delay: 0.2 }}
          viewport={{ once: true }}
          className="relative"
        >
          <div className="bg-slate-900/80 backdrop-blur-sm rounded-2xl border border-slate-700 shadow-2xl overflow-hidden">
            {/* Terminal Header */}
            <div className="flex items-center justify-between px-6 py-4 bg-slate-800 border-b border-slate-700">
              <div className="flex gap-2">
                <div className="w-3 h-3 bg-red-500 rounded-full"></div>
                <div className="w-3 h-3 bg-yellow-500 rounded-full"></div>
                <div className="w-3 h-3 bg-green-500 rounded-full"></div>
              </div>
              <span className="text-slate-400 text-sm font-mono">agent_example.py</span>
              <div className="flex items-center gap-2 text-green-400">
                <Activity className="w-4 h-4 animate-pulse" />
                <span className="text-xs">PRODUCTION</span>
              </div>
            </div>
            
            {/* Code Content */}
            <div className="p-8 font-mono text-sm overflow-x-auto">
              <pre className="text-slate-300">
<span className="text-purple-400">from</span> <span className="text-blue-400">src.core.agent_builder</span> <span className="text-purple-400">import</span> <span className="text-yellow-400">AgentBuilder</span>
<span className="text-purple-400">from</span> <span className="text-blue-400">langchain.agents</span> <span className="text-purple-400">import</span> <span className="text-yellow-400">tool</span>
<span className="text-purple-400">import</span> <span className="text-blue-400">os</span>

<span className="text-slate-500"># Define custom tools</span>
<span className="text-blue-400">@tool</span>
<span className="text-purple-400">def</span> <span className="text-yellow-400">analyze_sentiment</span>(<span className="text-orange-400">text</span>: <span className="text-green-400">str</span>) -&gt; <span className="text-green-400">str</span>:
    <span className="text-green-400">"""Analyze sentiment with memory context"""</span>
    <span className="text-purple-400">return</span> <span className="text-green-400">f"Sentiment: </span><span className="text-orange-400">&#123;text&#125;</span><span className="text-green-400"> -&gt; Positive"</span>

<span className="text-slate-500"># Create agent with 5-component memory</span>
<span className="text-orange-400">agent</span> = <span className="text-yellow-400">AgentBuilder</span>.<span className="text-yellow-400">create_agent</span>(
    <span className="text-orange-400">agent_name</span>=<span className="text-green-400">"production_assistant"</span>,
    <span className="text-orange-400">system_prompt</span>=<span className="text-green-400">"You are an AI with persistent memory."</span>,
    <span className="text-orange-400">user_tools</span>=[<span className="text-yellow-400">analyze_sentiment</span>],
    <span className="text-orange-400">mongodb_uri</span>=<span className="text-blue-400">os</span>.<span className="text-yellow-400">getenv</span>(<span className="text-green-400">"MONGODB_URI"</span>),
    <span className="text-orange-400">embedding_model</span>=<span className="text-green-400">"voyage-3-large"</span>
)

<span className="text-slate-500"># Agent remembers across sessions</span>
<span className="text-orange-400">response</span> = <span className="text-purple-400">await</span> <span className="text-orange-400">agent</span>.<span className="text-yellow-400">aexecute</span>(
    <span className="text-green-400">"Analyze: This framework is amazing!"</span>,
    <span className="text-orange-400">thread_id</span>=<span className="text-green-400">"user_session_123"</span>
)

<span className="text-slate-500"># Memory persists - ask follow-up</span>
<span className="text-orange-400">response</span> = <span className="text-purple-400">await</span> <span className="text-orange-400">agent</span>.<span className="text-yellow-400">aexecute</span>(
    <span className="text-green-400">"What was the sentiment of my last message?"</span>,
    <span className="text-orange-400">thread_id</span>=<span className="text-green-400">"user_session_123"</span>
)
<span className="text-yellow-400">print</span>(<span className="text-orange-400">response</span>)  <span className="text-slate-500"># Agent remembers the context</span>
              </pre>
            </div>
          </div>
        </motion.div>
      </div>
    </section>
  );
};

// Real Tech Stack with Glassmorphism
const TechStack = () => {
  const technologies = [
    {
      name: "LangGraph",
      description: "Stateful, multi-actor applications with LLMs",
      logo: <Workflow className="w-8 h-8" />,
      color: "from-blue-500 to-blue-600",
      features: ["State management", "Checkpointing", "Multi-agent workflows"]
    },
    {
      name: "MongoDB Atlas",
      description: "Vector search & document storage",
      logo: <Database className="w-8 h-8" />,
      color: "from-green-500 to-green-600", 
      features: ["Vector search", "Atlas Search", "Flexible schema"]
    },
    {
      name: "Voyage AI",
      description: "State-of-the-art embeddings",
      logo: <Zap className="w-8 h-8" />,
      color: "from-purple-500 to-purple-600",
      features: ["voyage-3-large", "1024 dimensions", "High accuracy"]
    },
    {
      name: "FastAPI",
      description: "High-performance API framework",
      logo: <Rocket className="w-8 h-8" />,
      color: "from-orange-500 to-orange-600",
      features: ["Async/await", "Auto docs", "Type hints"]
    },
    {
      name: "Docker",
      description: "Containerization & deployment",
      logo: <Shield className="w-8 h-8" />,
      color: "from-cyan-500 to-cyan-600",
      features: ["Multi-stage builds", "Production ready", "K8s compatible"]
    },
    {
      name: "Observability",
      description: "Monitoring & performance tracking",
      logo: <Activity className="w-8 h-8" />,
      color: "from-red-500 to-red-600",
      features: ["Galileo AI", "Metrics", "Tracing"]
    }
  ];
  
  return (
    <section className="py-24 bg-gradient-to-b from-slate-950 to-slate-900 relative">
      <div className="absolute inset-0 bg-grid-white/[0.02] bg-[size:50px_50px]" />
      
      <div className="max-w-7xl mx-auto px-6 relative">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true }}
          className="text-center mb-16"
        >
          <h2 className="text-4xl md:text-5xl font-black text-white mb-6">
            Production Stack
          </h2>
          <p className="text-xl text-slate-400 max-w-3xl mx-auto">
            Built with battle-tested technologies, not experimental frameworks
          </p>
        </motion.div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {technologies.map((tech, index) => (
            <motion.div
              key={tech.name}
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: index * 0.1 }}
              viewport={{ once: true }}
              className="group"
              whileHover={{ scale: 1.02 }}
            >
              <div className="bg-white/5 backdrop-blur-sm p-6 rounded-2xl border border-white/10 hover:border-blue-500/50 transition-all duration-300 h-full relative overflow-hidden">
                {/* Gradient overlay */}
                <div className={`absolute inset-0 bg-gradient-to-br ${tech.color} opacity-0 group-hover:opacity-10 transition-opacity duration-300`} />
                
                <div className={`inline-flex p-3 rounded-xl bg-gradient-to-r ${tech.color} text-white mb-4 relative`}>
                  {tech.logo}
                </div>
                <h3 className="text-xl font-bold text-white mb-2">{tech.name}</h3>
                <p className="text-slate-400 mb-4">{tech.description}</p>
                
                <ul className="space-y-1">
                  {tech.features.map((feature, i) => (
                    <li key={i} className="flex items-center gap-2 text-sm text-slate-500">
                      <CheckCircle className="w-4 h-4 text-green-500" />
                      {feature}
                    </li>
                  ))}
                </ul>
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
};

// Real CTA with Neural Design
const CallToAction = () => {
  return (
    <section className="py-24 bg-gradient-to-br from-slate-900 to-slate-950 relative">
      {/* Neural network background */}
      <div className="absolute inset-0 opacity-20">
        <NeuralNetwork />
      </div>
      
      <div className="max-w-4xl mx-auto px-6 text-center relative">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true }}
        >
          <h2 className="text-4xl md:text-5xl font-black text-white mb-6">
            Ready to Build?
          </h2>
          <p className="text-xl text-slate-300 mb-8 max-w-2xl mx-auto">
            Clone the repository and have a production-ready agent with memory running in 5 minutes.
          </p>
          
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <a
              href="https://github.com/romiluz13/agent_with_memory"
              className="inline-flex items-center gap-2 px-8 py-4 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-500 hover:to-purple-500 text-white font-semibold rounded-xl transition-all duration-300 hover:scale-105 hover:shadow-lg hover:shadow-purple-500/25"
            >
              <Github className="w-5 h-5" />
              Clone Repository
              <ExternalLink className="w-4 h-4" />
            </a>
            
            <a
              href="https://github.com/romiluz13/agent_with_memory"
              className="inline-flex items-center gap-2 px-8 py-4 bg-white/10 hover:bg-white/20 text-white font-semibold rounded-xl border border-white/20 transition-all duration-300 backdrop-blur-sm"
            >
              <Star className="w-5 h-5" />
              Star on GitHub
            </a>
          </div>
          
          <div className="mt-8 text-slate-400 text-sm">
            MIT License • Production Ready • 5-Minute Setup
          </div>
        </motion.div>
      </div>
    </section>
  );
};

export default function Home() {
  return (
    <>
      <Head>
        <title>AI Agent Boilerplate with Memory | LangGraph + MongoDB + Voyage AI</title>
        <meta name="description" content="Production-ready AI agent framework with 5-component memory system. Built on LangGraph, MongoDB Atlas, and Voyage AI. Deploy in 5 minutes." />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/favicon.ico" />
        <meta property="og:title" content="AI Agent Boilerplate with Memory" />
        <meta property="og:description" content="Production-ready agents with persistent memory in 5 minutes" />
        <meta property="og:type" content="website" />
        <meta name="twitter:card" content="summary_large_image" />
      </Head>

      <main className="font-display bg-slate-950">
        <Hero />
        <MemoryArchitecture />
        <CodeImplementation />
        <TechStack />
        <CallToAction />
      </main>
    </>
  );
}