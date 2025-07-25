import threading
import random
from collections import deque

class ThreadSafeReplayBuffer:
    def __init__(self, maxlen=100000):
        self.memory = deque(maxlen=maxlen)
        self.lock = threading.RLock()
    
    def push(self, state, action, reward, next_state, done):
        with self.lock:
            self.memory.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        with self.lock:
            if len(self.memory) < batch_size:
                return list(self.memory)
            return random.sample(self.memory, batch_size)
    
    def __len__(self):
        with self.lock:
            return len(self.memory)
    
    def clear(self):
        with self.lock:
            self.memory.clear()

class PrioritizedReplayBuffer:
    def __init__(self, maxlen=100000, alpha=0.6):
        self.memory = deque(maxlen=maxlen)
        self.priorities = deque(maxlen=maxlen)
        self.alpha = alpha
        self.lock = threading.RLock()
    
    def push(self, state, action, reward, next_state, done, priority=None):
        with self.lock:
            if priority is None:
                priority = max(self.priorities) if self.priorities else 1.0
            
            self.memory.append((state, action, reward, next_state, done))
            self.priorities.append(priority)
    
    def sample(self, batch_size, beta=0.4):
        with self.lock:
            if len(self.memory) < batch_size:
                return list(self.memory), [1.0] * len(self.memory), list(range(len(self.memory)))
            
            priorities = list(self.priorities)
            probs = [p ** self.alpha for p in priorities]
            probs = [p / sum(probs) for p in probs]
            
            indices = random.choices(range(len(self.memory)), weights=probs, k=batch_size)
            samples = [self.memory[i] for i in indices]
            
            weights = [(len(self.memory) * probs[i]) ** (-beta) for i in indices]
            max_weight = max(weights)
            weights = [w / max_weight for w in weights]
            
            return samples, weights, indices
    
    def update_priorities(self, indices, priorities):
        with self.lock:
            for idx, priority in zip(indices, priorities):
                if idx < len(self.priorities):
                    self.priorities[idx] = priority
    
    def __len__(self):
        with self.lock:
            return len(self.memory)