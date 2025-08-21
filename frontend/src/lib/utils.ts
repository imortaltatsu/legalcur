import { type ClassValue, clsx } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

// API base URL - using the legal.adityaberry.me domain
export const API_BASE_URL = 'https://legal.adityaberry.me'

// Log the API URL being used (helpful for debugging)
console.log('🔗 Using API URL:', API_BASE_URL)

// API types
export interface QueryRequest {
  query: string
  k?: number
}

export interface ChatMessage {
  role: 'system' | 'user' | 'assistant'
  content: string
}

export interface ChatRequest {
  messages: ChatMessage[]
  k?: number
}

export interface QueryResponse {
  answer: string
  sources: string[]
}

export interface SearchResponse {
  matches: Array<{
    source: string
    preview: string
  }>
}

export interface StatsResponse {
  collection: string
  doc_count: number
  bm25_ready: boolean
  corpus_file: string
  cases_dir: string
  statutes_dir: string
}

// API functions
export async function queryAPI(request: QueryRequest): Promise<QueryResponse> {
  const response = await fetch(`${API_BASE_URL}/query`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  })
  
  if (!response.ok) {
    throw new Error(`API request failed: ${response.statusText}`)
  }
  
  return response.json()
}

export async function searchAPI(request: QueryRequest): Promise<SearchResponse> {
  const response = await fetch(`${API_BASE_URL}/search`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  })
  
  if (!response.ok) {
    throw new Error(`API request failed: ${response.statusText}`)
  }
  
  return response.json()
}

export async function chatAPI(request: ChatRequest): Promise<QueryResponse> {
  const response = await fetch(`${API_BASE_URL}/chat`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  })
  
  if (!response.ok) {
    throw new Error(`API request failed: ${response.statusText}`)
  }
  
  return response.json()
}

export async function getStats(): Promise<StatsResponse> {
  const response = await fetch(`${API_BASE_URL}/stats`)
  
  if (!response.ok) {
    throw new Error(`API request failed: ${response.statusText}`)
  }
  
  return response.json()
}
