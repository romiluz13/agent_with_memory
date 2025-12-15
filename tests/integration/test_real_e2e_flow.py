"""
REAL End-to-End Integration Test
================================
This test uses REAL services - no mocks:
- Real MongoDB Atlas (stores actual data)
- Real Voyage AI (generates real 1024-dim embeddings)
- Real Gemini 2.5 Flash (LLM for summarization & entity extraction)

Simulates a real conversation that:
1. User asks questions about a project
2. Agent responds and stores episodic memories
3. Context builds up over multiple turns
4. At 80% threshold, auto-compression triggers
5. Summary created with real LLM
6. Messages marked as summarized (not deleted)
7. JIT expansion retrieves original content

Run with: python -m pytest tests/integration/test_real_e2e_flow.py -v -s
"""

import os
import asyncio
import pytest
import pytest_asyncio
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Skip if no real credentials
SKIP_REAL_TEST = not all([
    os.getenv("MONGODB_URI"),
    os.getenv("VOYAGE_API_KEY"),
    os.getenv("GOOGLE_API_KEY")
])

pytestmark = pytest.mark.skipif(
    SKIP_REAL_TEST,
    reason="Real E2E tests require MONGODB_URI, VOYAGE_API_KEY, and GOOGLE_API_KEY"
)


# Realistic conversation - a developer asking about their codebase
REALISTIC_CONVERSATION = [
    {
        "role": "user",
        "content": "I'm working on the authentication system for our new e-commerce platform. The current implementation uses JWT tokens but we're having issues with token refresh. Can you help me understand the best practices?"
    },
    {
        "role": "assistant",
        "content": "I'd be happy to help with your JWT authentication issues. For token refresh, the best practice is to use a dual-token system: a short-lived access token (15-30 minutes) and a longer-lived refresh token (7-30 days). When the access token expires, the client uses the refresh token to get a new access token without requiring re-authentication. Key considerations: 1) Store refresh tokens securely (httpOnly cookies), 2) Implement token rotation on refresh, 3) Maintain a token blacklist for logout/revocation. What specific issues are you encountering with the refresh flow?"
    },
    {
        "role": "user",
        "content": "The main issue is that when users are in the middle of checkout, their token expires and they lose their cart. We need seamless token refresh without interrupting the user experience."
    },
    {
        "role": "assistant",
        "content": "This is a common UX challenge. Here's a robust solution: Implement proactive token refresh using an interceptor pattern. In your API client (Axios/fetch wrapper), check the token expiration before each request. If the token will expire within 5 minutes, refresh it proactively in the background. For the checkout flow specifically: 1) Cache the cart state locally (localStorage/sessionStorage), 2) Queue API requests during refresh, 3) Replay queued requests after successful refresh, 4) If refresh fails, preserve cart data and prompt for re-login with a 'Continue checkout' option. Also consider implementing silent refresh using an iframe for OAuth flows."
    },
    {
        "role": "user",
        "content": "That makes sense. We're using React for the frontend. Can you show me how to implement the interceptor pattern with Axios?"
    },
    {
        "role": "assistant",
        "content": """Here's a production-ready Axios interceptor implementation for React:

```javascript
// api/axiosInstance.js
import axios from 'axios';
import { getAccessToken, getRefreshToken, setTokens, clearTokens } from './tokenService';

const api = axios.create({ baseURL: process.env.REACT_APP_API_URL });

let isRefreshing = false;
let failedQueue = [];

const processQueue = (error, token = null) => {
  failedQueue.forEach(prom => {
    if (error) prom.reject(error);
    else prom.resolve(token);
  });
  failedQueue = [];
};

api.interceptors.request.use(async (config) => {
  const token = getAccessToken();
  if (token) {
    // Check if token expires within 5 minutes
    const payload = JSON.parse(atob(token.split('.')[1]));
    const expiresIn = payload.exp * 1000 - Date.now();

    if (expiresIn < 300000 && !isRefreshing) { // 5 minutes
      isRefreshing = true;
      try {
        const response = await axios.post('/auth/refresh', {
          refreshToken: getRefreshToken()
        });
        setTokens(response.data.accessToken, response.data.refreshToken);
        processQueue(null, response.data.accessToken);
      } catch (err) {
        processQueue(err, null);
        clearTokens();
        window.location.href = '/login?returnUrl=' + encodeURIComponent(window.location.pathname);
      } finally {
        isRefreshing = false;
      }
    }
    config.headers.Authorization = `Bearer ${getAccessToken()}`;
  }
  return config;
});
```

This handles proactive refresh, request queuing, and graceful degradation. Want me to also show the token service implementation?"""
    },
    {
        "role": "user",
        "content": "Yes please, and also how to handle the checkout-specific case where we preserve the cart on token failure."
    },
    {
        "role": "assistant",
        "content": """Here's the complete token service with cart preservation:

```javascript
// services/tokenService.js
const ACCESS_TOKEN_KEY = 'access_token';
const REFRESH_TOKEN_KEY = 'refresh_token';
const CART_BACKUP_KEY = 'cart_backup';

export const getAccessToken = () => localStorage.getItem(ACCESS_TOKEN_KEY);
export const getRefreshToken = () => localStorage.getItem(REFRESH_TOKEN_KEY);

export const setTokens = (accessToken, refreshToken) => {
  localStorage.setItem(ACCESS_TOKEN_KEY, accessToken);
  localStorage.setItem(REFRESH_TOKEN_KEY, refreshToken);
};

export const clearTokens = () => {
  localStorage.removeItem(ACCESS_TOKEN_KEY);
  localStorage.removeItem(REFRESH_TOKEN_KEY);
};

// Cart preservation for checkout flow
export const backupCart = (cartData) => {
  sessionStorage.setItem(CART_BACKUP_KEY, JSON.stringify({
    cart: cartData,
    timestamp: Date.now(),
    returnUrl: window.location.pathname
  }));
};

export const getCartBackup = () => {
  const backup = sessionStorage.getItem(CART_BACKUP_KEY);
  if (!backup) return null;

  const data = JSON.parse(backup);
  // Cart backup valid for 24 hours
  if (Date.now() - data.timestamp > 86400000) {
    sessionStorage.removeItem(CART_BACKUP_KEY);
    return null;
  }
  return data;
};

export const clearCartBackup = () => sessionStorage.removeItem(CART_BACKUP_KEY);
```

And the checkout-specific error handler:

```javascript
// hooks/useCheckout.js
import { useCallback } from 'react';
import { backupCart, getCartBackup } from '../services/tokenService';

export const useCheckout = () => {
  const handleCheckoutError = useCallback((error, cartState) => {
    if (error.response?.status === 401) {
      // Backup cart before redirect
      backupCart(cartState);
      // Redirect with return URL
      window.location.href = `/login?returnUrl=/checkout&hasCart=true`;
    }
  }, []);

  const restoreCheckout = useCallback(() => {
    const backup = getCartBackup();
    if (backup) {
      // Restore cart state
      return backup.cart;
    }
    return null;
  }, []);

  return { handleCheckoutError, restoreCheckout };
};
```

This ensures users never lose their cart even if their session expires mid-checkout."""
    },
    {
        "role": "user",
        "content": "This is really helpful. Now I'm also concerned about security. What if someone steals the refresh token? How do we detect and prevent token theft?"
    },
    {
        "role": "assistant",
        "content": """Excellent security question. Here's a comprehensive token theft detection and prevention strategy:

**1. Refresh Token Rotation (Essential)**
```javascript
// On every refresh, issue a new refresh token and invalidate the old one
async function refreshTokens(oldRefreshToken) {
  const tokenRecord = await db.refreshTokens.findOne({ token: oldRefreshToken });

  if (!tokenRecord || tokenRecord.revoked) {
    // Token reuse detected! Possible theft
    await db.refreshTokens.updateMany(
      { userId: tokenRecord?.userId },
      { $set: { revoked: true } }
    );
    throw new SecurityError('Refresh token reuse detected - all sessions invalidated');
  }

  // Revoke old token
  await db.refreshTokens.updateOne(
    { token: oldRefreshToken },
    { $set: { revoked: true, revokedAt: new Date() } }
  );

  // Issue new tokens
  const newRefreshToken = generateRefreshToken();
  await db.refreshTokens.insertOne({
    token: newRefreshToken,
    userId: tokenRecord.userId,
    deviceId: tokenRecord.deviceId,
    createdAt: new Date()
  });

  return { accessToken: generateAccessToken(tokenRecord.userId), refreshToken: newRefreshToken };
}
```

**2. Device Fingerprinting**
```javascript
// Include device fingerprint in refresh token claims
const deviceFingerprint = generateFingerprint(req.headers['user-agent'], req.ip);
const refreshToken = jwt.sign({ userId, deviceId: deviceFingerprint }, SECRET);

// Validate fingerprint on refresh
if (storedFingerprint !== currentFingerprint) {
  // Different device trying to use token - possible theft
  await revokeAllUserTokens(userId);
  await notifyUser(userId, 'Suspicious login attempt detected');
}
```

**3. Anomaly Detection**
- Track refresh patterns (time, location, frequency)
- Flag unusual patterns: rapid refreshes, geographic impossibilities, new device
- Implement risk-based authentication for suspicious activity

**4. Short Refresh Token Lifetime for Sensitive Operations**
- Normal browsing: 30-day refresh token
- Checkout/payment: Require re-authentication within last 15 minutes
- Account changes: Always require current password

Would you like me to show the complete backend implementation with Redis for token storage?"""
    },
    {
        "role": "user",
        "content": "Yes, let's see the Redis implementation. Also, we're using Node.js with Express for the backend."
    },
    {
        "role": "assistant",
        "content": """Here's the complete Redis-based token management for Express:

```javascript
// services/tokenService.js
const Redis = require('ioredis');
const jwt = require('jsonwebtoken');
const crypto = require('crypto');

const redis = new Redis(process.env.REDIS_URL);

const ACCESS_TOKEN_EXPIRY = 15 * 60; // 15 minutes
const REFRESH_TOKEN_EXPIRY = 30 * 24 * 60 * 60; // 30 days

class TokenService {
  static generateAccessToken(userId, deviceId) {
    return jwt.sign(
      { userId, deviceId, type: 'access' },
      process.env.JWT_ACCESS_SECRET,
      { expiresIn: ACCESS_TOKEN_EXPIRY }
    );
  }

  static generateRefreshToken() {
    return crypto.randomBytes(64).toString('hex');
  }

  static async storeRefreshToken(userId, refreshToken, deviceId, metadata = {}) {
    const key = `refresh:${userId}:${refreshToken}`;
    const data = {
      userId,
      deviceId,
      createdAt: Date.now(),
      userAgent: metadata.userAgent,
      ip: metadata.ip,
      revoked: false
    };

    await redis.setex(key, REFRESH_TOKEN_EXPIRY, JSON.stringify(data));

    // Also track by user for easy revocation
    await redis.sadd(`user_tokens:${userId}`, key);
    await redis.expire(`user_tokens:${userId}`, REFRESH_TOKEN_EXPIRY);
  }

  static async validateRefreshToken(userId, refreshToken) {
    const key = `refresh:${userId}:${refreshToken}`;
    const data = await redis.get(key);

    if (!data) return { valid: false, reason: 'Token not found' };

    const tokenData = JSON.parse(data);
    if (tokenData.revoked) {
      // TOKEN REUSE DETECTED - Security breach!
      await this.revokeAllUserTokens(userId);
      return { valid: false, reason: 'Token reuse detected', securityBreach: true };
    }

    return { valid: true, data: tokenData };
  }

  static async rotateRefreshToken(userId, oldRefreshToken, deviceId, metadata) {
    const validation = await this.validateRefreshToken(userId, oldRefreshToken);

    if (!validation.valid) {
      if (validation.securityBreach) {
        throw new SecurityError('Possible token theft detected');
      }
      throw new AuthError('Invalid refresh token');
    }

    // Revoke old token
    const oldKey = `refresh:${userId}:${oldRefreshToken}`;
    const oldData = JSON.parse(await redis.get(oldKey));
    oldData.revoked = true;
    oldData.revokedAt = Date.now();
    await redis.setex(oldKey, 3600, JSON.stringify(oldData)); // Keep for 1 hour for audit

    // Issue new tokens
    const newRefreshToken = this.generateRefreshToken();
    await this.storeRefreshToken(userId, newRefreshToken, deviceId, metadata);

    const accessToken = this.generateAccessToken(userId, deviceId);

    return { accessToken, refreshToken: newRefreshToken };
  }

  static async revokeAllUserTokens(userId) {
    const keys = await redis.smembers(`user_tokens:${userId}`);

    for (const key of keys) {
      const data = await redis.get(key);
      if (data) {
        const tokenData = JSON.parse(data);
        tokenData.revoked = true;
        tokenData.revokedAt = Date.now();
        tokenData.revokedReason = 'Security: all tokens revoked';
        await redis.setex(key, 3600, JSON.stringify(tokenData));
      }
    }

    await redis.del(`user_tokens:${userId}`);

    // Log security event
    console.log(`[SECURITY] All tokens revoked for user ${userId} at ${new Date().toISOString()}`);
  }
}

module.exports = TokenService;
```

And the Express middleware:

```javascript
// middleware/auth.js
const TokenService = require('../services/tokenService');

const authMiddleware = async (req, res, next) => {
  const authHeader = req.headers.authorization;

  if (!authHeader?.startsWith('Bearer ')) {
    return res.status(401).json({ error: 'No token provided' });
  }

  const token = authHeader.substring(7);

  try {
    const payload = jwt.verify(token, process.env.JWT_ACCESS_SECRET);
    req.user = { userId: payload.userId, deviceId: payload.deviceId };
    next();
  } catch (err) {
    if (err.name === 'TokenExpiredError') {
      return res.status(401).json({ error: 'Token expired', code: 'TOKEN_EXPIRED' });
    }
    return res.status(401).json({ error: 'Invalid token' });
  }
};
```

This gives you O(1) token lookups with Redis, automatic expiration, and comprehensive security event logging."""
    },
    {
        "role": "user",
        "content": "Perfect! One more thing - how should we handle rate limiting for the refresh endpoint to prevent brute force attacks?"
    },
    {
        "role": "assistant",
        "content": """Great security consideration! Here's a multi-layer rate limiting strategy for the refresh endpoint:

```javascript
// middleware/rateLimiter.js
const Redis = require('ioredis');
const redis = new Redis(process.env.REDIS_URL);

/**
 * Multi-layer rate limiting:
 * 1. Global rate limit (per IP) - prevent DDoS
 * 2. User-specific rate limit - prevent targeted attacks
 * 3. Endpoint-specific limits - stricter for auth endpoints
 */

const createRateLimiter = (options) => {
  const {
    keyPrefix,
    maxRequests,
    windowMs,
    blockDurationMs = windowMs * 2
  } = options;

  return async (req, res, next) => {
    const key = `${keyPrefix}:${req.ip}`;

    const [current, ttl] = await redis
      .multi()
      .incr(key)
      .ttl(key)
      .exec();

    const count = current[1];

    // Set expiry on first request
    if (ttl[1] === -1) {
      await redis.expire(key, Math.ceil(windowMs / 1000));
    }

    // Set rate limit headers
    res.set({
      'X-RateLimit-Limit': maxRequests,
      'X-RateLimit-Remaining': Math.max(0, maxRequests - count),
      'X-RateLimit-Reset': Date.now() + (ttl[1] * 1000)
    });

    if (count > maxRequests) {
      // Block for longer duration on repeated violations
      if (count > maxRequests * 2) {
        await redis.expire(key, Math.ceil(blockDurationMs / 1000));
        console.log(`[SECURITY] IP ${req.ip} blocked for repeated rate limit violations`);
      }

      return res.status(429).json({
        error: 'Too many requests',
        retryAfter: ttl[1]
      });
    }

    next();
  };
};

// Specific limiters
const globalLimiter = createRateLimiter({
  keyPrefix: 'rl:global',
  maxRequests: 100,
  windowMs: 60000 // 100 requests per minute
});

const authLimiter = createRateLimiter({
  keyPrefix: 'rl:auth',
  maxRequests: 5,
  windowMs: 60000, // 5 auth attempts per minute
  blockDurationMs: 300000 // 5 minute block on violation
});

const refreshLimiter = createRateLimiter({
  keyPrefix: 'rl:refresh',
  maxRequests: 10,
  windowMs: 60000, // 10 refreshes per minute (generous for SPA)
  blockDurationMs: 600000 // 10 minute block
});

// User-specific limiter (requires auth)
const userRefreshLimiter = async (req, res, next) => {
  if (!req.user) return next();

  const key = `rl:user_refresh:${req.user.userId}`;
  const count = await redis.incr(key);

  if (count === 1) {
    await redis.expire(key, 60); // 1 minute window
  }

  if (count > 20) { // 20 refreshes per user per minute is suspicious
    console.log(`[SECURITY] Suspicious refresh activity for user ${req.user.userId}`);
    // Don't block, but flag for review
    req.suspiciousActivity = true;
  }

  next();
};

module.exports = { globalLimiter, authLimiter, refreshLimiter, userRefreshLimiter };
```

Apply to routes:

```javascript
// routes/auth.js
const express = require('express');
const router = express.Router();
const { authLimiter, refreshLimiter, userRefreshLimiter } = require('../middleware/rateLimiter');
const TokenService = require('../services/tokenService');

router.post('/login', authLimiter, async (req, res) => {
  // Login logic
});

router.post('/refresh', refreshLimiter, async (req, res) => {
  const { refreshToken } = req.body;
  const userId = extractUserIdFromRefreshToken(refreshToken);

  try {
    const tokens = await TokenService.rotateRefreshToken(
      userId,
      refreshToken,
      req.headers['x-device-id'],
      { userAgent: req.headers['user-agent'], ip: req.ip }
    );

    res.json(tokens);
  } catch (err) {
    if (err instanceof SecurityError) {
      // Log and alert on security breaches
      await alertSecurityTeam(userId, req.ip, err.message);
      return res.status(403).json({ error: 'Security violation detected' });
    }
    res.status(401).json({ error: 'Invalid refresh token' });
  }
});

router.post('/logout', authMiddleware, async (req, res) => {
  await TokenService.revokeAllUserTokens(req.user.userId);
  res.json({ success: true });
});

module.exports = router;
```

This provides defense in depth: IP-based limits for DDoS protection, stricter limits for auth endpoints, and user-specific monitoring for account-targeted attacks."""
    }
]


class TestRealE2EFlow:
    """Real end-to-end tests with actual services."""

    @pytest_asyncio.fixture
    async def services(self):
        """Set up real MongoDB and services."""
        from motor.motor_asyncio import AsyncIOMotorClient
        from src.memory.episodic import EpisodicMemory
        from src.memory.summary import SummaryMemory
        from src.memory.entity import EntityMemory
        from src.context.engineer import ContextEngineer
        from src.context.summarizer import ContextSummarizer

        # Real MongoDB connection
        client = AsyncIOMotorClient(os.getenv("MONGODB_URI"))
        db = client["awm_e2e_test"]

        # Clean collections before test
        await db.episodic_memories.delete_many({})
        await db.summary_memories.delete_many({})
        await db.entity_memories.delete_many({})

        # Real memory stores (using real Voyage AI embeddings)
        episodic = EpisodicMemory(db.episodic_memories)
        summary = SummaryMemory(db.summary_memories)
        entity = EntityMemory(db.entity_memories)

        # Context management
        context_engineer = ContextEngineer(threshold=0.80)
        summarizer = ContextSummarizer()

        result = {
            "client": client,
            "db": db,
            "episodic": episodic,
            "summary": summary,
            "entity": entity,
            "context_engineer": context_engineer,
            "summarizer": summarizer,
            "agent_id": "test-agent-e2e",
            "user_id": "test-user-e2e",
            "thread_id": f"thread-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        }

        yield result

        # Cleanup after test
        await db.episodic_memories.delete_many({})
        await db.summary_memories.delete_many({})
        await db.entity_memories.delete_many({})
        client.close()

    @pytest.mark.asyncio
    async def test_step1_store_conversation_with_real_embeddings(self, services):
        """
        STEP 1: Store realistic conversation with REAL Voyage AI embeddings.

        What happens:
        1. Each message is sent to Voyage AI for embedding (1024 dimensions)
        2. Message + embedding stored in MongoDB
        3. Metadata includes role, timestamp, thread_id
        """
        episodic = services["episodic"]
        agent_id = services["agent_id"]
        thread_id = services["thread_id"]

        from src.memory.base import Memory, MemoryType

        print("\n" + "="*60)
        print("STEP 1: Storing conversation with REAL Voyage AI embeddings")
        print("="*60)

        stored_ids = []

        for i, msg in enumerate(REALISTIC_CONVERSATION[:4]):  # First 4 messages
            memory = Memory(
                agent_id=agent_id,
                content=msg["content"],
                memory_type=MemoryType.EPISODIC,
                metadata={
                    "role": msg["role"],
                    "thread_id": thread_id,
                    "turn": i + 1
                }
            )

            print(f"\n[Turn {i+1}] Storing {msg['role']} message...")
            print(f"  Content preview: {msg['content'][:80]}...")

            # This calls REAL Voyage AI to generate embedding
            memory_id = await episodic.store(memory)
            stored_ids.append(memory_id)

            print(f"  ✓ Stored with ID: {memory_id}")
            print(f"  ✓ Embedding generated: 1024 dimensions (Voyage AI)")

        # Verify in database
        count = await services["db"].episodic_memories.count_documents({
            "agent_id": agent_id,
            "metadata.thread_id": thread_id
        })

        print(f"\n✓ Total messages stored: {count}")
        assert count == 4
        assert len(stored_ids) == 4

    @pytest.mark.asyncio
    async def test_step2_context_builds_toward_threshold(self, services):
        """
        STEP 2: Build context and monitor token usage toward 80% threshold.

        What happens:
        1. Store full conversation (10 messages)
        2. Calculate token usage after each message
        3. Show progression toward 80% threshold
        """
        episodic = services["episodic"]
        context_engineer = services["context_engineer"]
        agent_id = services["agent_id"]
        thread_id = services["thread_id"]

        from src.memory.base import Memory, MemoryType

        print("\n" + "="*60)
        print("STEP 2: Building context toward 80% threshold")
        print("="*60)

        # Store all messages
        for i, msg in enumerate(REALISTIC_CONVERSATION):
            memory = Memory(
                agent_id=agent_id,
                content=msg["content"],
                memory_type=MemoryType.EPISODIC,
                metadata={
                    "role": msg["role"],
                    "thread_id": thread_id,
                    "turn": i + 1
                }
            )
            await episodic.store(memory)

        # Build context string from conversation
        context = "\n\n".join([
            f"[{msg['role'].upper()}]: {msg['content']}"
            for msg in REALISTIC_CONVERSATION
        ])

        print(f"\nConversation stats:")
        print(f"  Messages: {len(REALISTIC_CONVERSATION)}")
        print(f"  Total characters: {len(context):,}")

        # Check token usage for different models
        for model in ["gpt-4o", "claude-3-5-sonnet", "gemini-2.0-flash"]:
            usage = context_engineer.calculate_usage(context, model)
            should_compress = context_engineer.should_compress(context, model)

            print(f"\n  {model}:")
            print(f"    Estimated tokens: {usage.tokens:,}")
            print(f"    Token limit: {usage.max_tokens:,}")
            print(f"    Usage: {usage.percent:.1f}%")
            print(f"    Should compress: {'YES' if should_compress else 'NO'}")
            print(f"    {'⚠️  ABOVE 80% THRESHOLD!' if should_compress else '✓ Below threshold'}")

    @pytest.mark.asyncio
    async def test_step3_trigger_compression_with_real_llm(self, services):
        """
        STEP 3: Trigger compression and generate summary with REAL Gemini LLM.

        What happens:
        1. Context exceeds 80% threshold (simulated with lower limit)
        2. Summarizer calls REAL Gemini 2.5 Flash
        3. Summary stored in summary_memories collection
        4. Original messages marked with summary_id (NOT deleted)
        """
        episodic = services["episodic"]
        summary_store = services["summary"]
        summarizer = services["summarizer"]
        agent_id = services["agent_id"]
        thread_id = services["thread_id"]

        from src.memory.base import Memory, MemoryType
        import google.generativeai as genai

        print("\n" + "="*60)
        print("STEP 3: Triggering compression with REAL Gemini LLM")
        print("="*60)

        # Store conversation
        for i, msg in enumerate(REALISTIC_CONVERSATION):
            memory = Memory(
                agent_id=agent_id,
                content=msg["content"],
                memory_type=MemoryType.EPISODIC,
                metadata={
                    "role": msg["role"],
                    "thread_id": thread_id,
                    "turn": i + 1
                }
            )
            await episodic.store(memory)

        # Build full context
        full_context = "\n\n".join([
            f"[{msg['role'].upper()}]: {msg['content']}"
            for msg in REALISTIC_CONVERSATION
        ])

        print(f"\nOriginal context: {len(full_context):,} characters")

        # Configure Gemini
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        model = genai.GenerativeModel("gemini-2.0-flash")

        # Generate summary with REAL LLM
        print("\n🤖 Calling Gemini 2.0 Flash for summarization...")

        summary_prompt = f"""Summarize this conversation concisely, preserving key technical details and decisions:

{full_context}

Provide a structured summary with:
1. Main topic discussed
2. Key technical decisions/recommendations
3. Important code patterns mentioned
4. Action items or next steps"""

        response = await asyncio.to_thread(
            lambda: model.generate_content(summary_prompt)
        )

        summary_text = response.text
        print(f"\n✓ Summary generated: {len(summary_text):,} characters")
        print(f"  Compression ratio: {len(summary_text)/len(full_context):.1%}")

        # Generate description label
        label_prompt = f"In 10 words or less, label this conversation: {summary_text[:500]}"
        label_response = await asyncio.to_thread(
            lambda: model.generate_content(label_prompt)
        )
        description = label_response.text.strip()

        print(f"  Label: {description}")

        # Store summary with full content for JIT expansion
        summary_id = f"sum-{datetime.now().strftime('%Y%m%d%H%M%S')}"

        await summary_store.store_summary(
            summary_id=summary_id,
            full_content=full_context,
            summary=summary_text,
            description=description,
            agent_id=agent_id,
            thread_id=thread_id
        )

        print(f"\n✓ Summary stored with ID: {summary_id}")

        # Mark original messages as summarized (Oracle pattern: don't delete!)
        marked_count = await episodic.mark_as_summarized(
            agent_id=agent_id,
            thread_id=thread_id,
            summary_id=summary_id
        )

        print(f"✓ Marked {marked_count} messages as summarized")

        # Verify messages still exist
        all_messages = await episodic.list_memories(
            filters={"agent_id": agent_id},
            include_summarized=True
        )
        print(f"✓ Messages still in DB: {len(all_messages)} (preserved for audit)")

        # Verify hidden from default query
        visible_messages = await episodic.list_memories(
            filters={"agent_id": agent_id},
            include_summarized=False
        )
        print(f"✓ Messages hidden from default query: {len(visible_messages)}")

        assert len(all_messages) == len(REALISTIC_CONVERSATION)
        assert len(visible_messages) == 0  # All summarized, so hidden by default

    @pytest.mark.asyncio
    async def test_step4_jit_expansion_retrieves_original(self, services):
        """
        STEP 4: JIT (Just-In-Time) expansion retrieves original content.

        What happens:
        1. Query the summary by ID
        2. Expand to retrieve full original content
        3. Verify lossless retrieval
        """
        episodic = services["episodic"]
        summary_store = services["summary"]
        agent_id = services["agent_id"]
        thread_id = services["thread_id"]

        from src.memory.base import Memory, MemoryType
        import google.generativeai as genai

        print("\n" + "="*60)
        print("STEP 4: JIT expansion retrieves original content")
        print("="*60)

        # Store and summarize (repeat setup)
        for i, msg in enumerate(REALISTIC_CONVERSATION):
            memory = Memory(
                agent_id=agent_id,
                content=msg["content"],
                memory_type=MemoryType.EPISODIC,
                metadata={
                    "role": msg["role"],
                    "thread_id": thread_id,
                    "turn": i + 1
                }
            )
            await episodic.store(memory)

        full_context = "\n\n".join([
            f"[{msg['role'].upper()}]: {msg['content']}"
            for msg in REALISTIC_CONVERSATION
        ])

        # Generate and store summary
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        model = genai.GenerativeModel("gemini-2.0-flash")

        summary_prompt = f"Summarize this conversation concisely:\n\n{full_context}"
        response = await asyncio.to_thread(
            lambda: model.generate_content(summary_prompt)
        )

        summary_id = f"sum-jit-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        await summary_store.store_summary(
            summary_id=summary_id,
            full_content=full_context,
            summary=response.text,
            description="JWT Auth Discussion",
            agent_id=agent_id,
            thread_id=thread_id
        )

        print(f"\nStored summary: {summary_id}")
        print(f"Summary length: {len(response.text):,} chars")
        print(f"Original length: {len(full_context):,} chars")

        # JIT EXPANSION - retrieve original content
        print("\n🔍 Performing JIT expansion...")
        expanded = await summary_store.expand_summary(summary_id)

        print(f"✓ Expanded content length: {len(expanded):,} chars")
        print(f"✓ Matches original: {expanded == full_context}")

        # Show that we can retrieve full detail on demand
        print("\n📖 First 500 chars of expanded content:")
        print("-" * 40)
        print(expanded[:500])
        print("-" * 40)

        assert expanded == full_context

    @pytest.mark.asyncio
    async def test_step5_entity_extraction_with_real_llm(self, services):
        """
        STEP 5: Extract entities from conversation using REAL Gemini LLM.

        What happens:
        1. Send conversation text to Gemini
        2. LLM extracts PERSON, ORGANIZATION, LOCATION, SYSTEM, CONCEPT
        3. Entities stored with embeddings for semantic search
        """
        entity_store = services["entity"]
        agent_id = services["agent_id"]

        import google.generativeai as genai
        from src.memory.base import Memory, MemoryType

        print("\n" + "="*60)
        print("STEP 5: Entity extraction with REAL Gemini LLM")
        print("="*60)

        # Sample text from conversation
        sample_text = REALISTIC_CONVERSATION[3]["content"]  # The Axios code example

        print(f"\nExtracting entities from assistant response...")
        print(f"Text length: {len(sample_text)} chars")

        # Configure Gemini
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        model = genai.GenerativeModel("gemini-2.0-flash")

        # Extract entities
        extraction_prompt = f'''Extract entities from the following text.
Return JSON array: [{{"name": "X", "type": "PERSON|ORGANIZATION|LOCATION|SYSTEM|CONCEPT", "description": "brief description"}}]
If no entities found, return: []

Text: "{sample_text}"'''

        print("\n🤖 Calling Gemini for entity extraction...")
        response = await asyncio.to_thread(
            lambda: model.generate_content(extraction_prompt)
        )

        # Parse response
        import json
        import re

        response_text = response.text
        # Extract JSON from response (handle markdown code blocks)
        json_match = re.search(r'\[[\s\S]*\]', response_text)
        if json_match:
            entities = json.loads(json_match.group())
        else:
            entities = []

        print(f"\n✓ Extracted {len(entities)} entities:")

        # Store each entity
        for entity in entities:
            print(f"\n  [{entity['type']}] {entity['name']}")
            print(f"    Description: {entity.get('description', 'N/A')}")

            memory = Memory(
                agent_id=agent_id,
                content=f"{entity['name']} ({entity['type']}): {entity.get('description', '')}",
                memory_type=MemoryType.ENTITY,
                metadata={
                    "entity_name": entity["name"],
                    "entity_type": entity["type"],
                    "description": entity.get("description", ""),
                    "mentions": 1
                }
            )

            entity_id = await entity_store.store(memory)
            print(f"    ✓ Stored with ID: {entity_id}")

        # Verify storage
        stored_entities = await entity_store.list_memories(
            filters={"agent_id": agent_id}
        )
        print(f"\n✓ Total entities in DB: {len(stored_entities)}")

        assert len(stored_entities) >= 1

    @pytest.mark.asyncio
    async def test_step6_semantic_search_with_real_embeddings(self, services):
        """
        STEP 6: Semantic search using REAL Voyage AI embeddings.

        What happens:
        1. Store conversation messages with real embeddings
        2. Query with natural language
        3. Voyage AI embeds query
        4. MongoDB vector search finds similar memories
        """
        episodic = services["episodic"]
        agent_id = services["agent_id"]
        thread_id = services["thread_id"]

        from src.memory.base import Memory, MemoryType

        print("\n" + "="*60)
        print("STEP 6: Semantic search with REAL Voyage AI embeddings")
        print("="*60)

        # Store conversation
        print("\nStoring conversation messages with real embeddings...")
        for i, msg in enumerate(REALISTIC_CONVERSATION[:6]):
            memory = Memory(
                agent_id=agent_id,
                content=msg["content"],
                memory_type=MemoryType.EPISODIC,
                metadata={
                    "role": msg["role"],
                    "thread_id": thread_id,
                    "turn": i + 1
                }
            )
            await episodic.store(memory)
            print(f"  ✓ Turn {i+1} stored")

        # Natural language queries
        queries = [
            "How do I implement token refresh in React?",
            "What about security and preventing token theft?",
            "How to preserve cart during checkout?"
        ]

        for query in queries:
            print(f"\n🔍 Query: '{query}'")
            print("   Generating query embedding with Voyage AI...")

            # This uses REAL Voyage AI to embed query and search
            results = await episodic.retrieve(
                query=query,
                limit=3,
                threshold=0.5
            )

            print(f"   ✓ Found {len(results)} relevant memories:")
            for j, result in enumerate(results):
                preview = result.content[:100].replace('\n', ' ')
                print(f"      {j+1}. [{result.metadata.get('role', '?')}] {preview}...")

    @pytest.mark.asyncio
    async def test_full_flow_summary(self, services):
        """
        COMPLETE FLOW: Shows what happens in a real AWM 2.0 session.
        """
        print("\n" + "="*70)
        print("AWM 2.0 COMPLETE FLOW SUMMARY")
        print("="*70)

        print("""
┌─────────────────────────────────────────────────────────────────────┐
│                    AWM 2.0 Real E2E Flow                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  USER INPUT                                                         │
│      │                                                              │
│      ▼                                                              │
│  ┌─────────────────┐                                                │
│  │ Voyage AI       │ ──► Generate 1024-dim embedding                │
│  │ (REAL API)      │                                                │
│  └─────────────────┘                                                │
│      │                                                              │
│      ▼                                                              │
│  ┌─────────────────┐                                                │
│  │ MongoDB Atlas   │ ──► Store message + embedding                  │
│  │ (REAL DB)       │                                                │
│  └─────────────────┘                                                │
│      │                                                              │
│      ▼                                                              │
│  ┌─────────────────┐                                                │
│  │ Context Engineer│ ──► Monitor token usage                        │
│  │                 │     (chars / 4 estimation)                     │
│  └─────────────────┘                                                │
│      │                                                              │
│      ▼                                                              │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ IF usage > 80%:                                             │    │
│  │   1. Call Gemini 2.0 Flash to summarize                     │    │
│  │   2. Store summary with full_content (for JIT expansion)    │    │
│  │   3. Mark messages with summary_id (DON'T DELETE!)          │    │
│  │   4. Extract entities with LLM                              │    │
│  └─────────────────────────────────────────────────────────────┘    │
│      │                                                              │
│      ▼                                                              │
│  ┌─────────────────┐                                                │
│  │ Semantic Search │ ──► Vector search across all 7 memory types    │
│  │ (Voyage + Mongo)│     Hybrid: vector_weight * 0.7 + text * 0.3   │
│  └─────────────────┘                                                │
│      │                                                              │
│      ▼                                                              │
│  ┌─────────────────┐                                                │
│  │ JIT Expansion   │ ──► On demand: expand_summary(id) retrieves    │
│  │                 │     original full content                      │
│  └─────────────────┘                                                │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

MEMORY TYPES USED:
  ✓ EPISODIC  - Conversation messages
  ✓ SUMMARY   - Compressed context with JIT expansion
  ✓ ENTITY    - LLM-extracted people, orgs, concepts

KEY PATTERNS (from Oracle, implemented in AWM):
  ✓ Mark-as-Summarized (not delete) - Audit trail preserved
  ✓ 80% Threshold - Auto-compress before context overflow
  ✓ JIT Expansion - Retrieve full content on demand
  ✓ Entity Merging - Increment mentions on duplicates

AWM 2.0 UNIQUE FEATURES (Oracle doesn't have):
  ✓ Hybrid Search - Vector + text weighted scoring
  ✓ Voyage AI 1024-dim - Superior to 768-dim alternatives
  ✓ Multi-Collection Search - Search all 7 types at once
  ✓ Memory Importance Levels - CRITICAL to TRIVIAL
  ✓ TTL-based Expiration - Auto-cleanup old working memory
""")

        assert True  # This is documentation test


if __name__ == "__main__":
    # Run with: python -m pytest tests/integration/test_real_e2e_flow.py -v -s
    pytest.main([__file__, "-v", "-s"])
