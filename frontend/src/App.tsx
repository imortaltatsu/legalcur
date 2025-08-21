import React, { useState, useEffect } from 'react'
import { Tabs, TabsContent, TabsList, TabsTrigger } from './components/ui/tabs'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './components/ui/card'
import { Button } from './components/ui/button'
import { Input } from './components/ui/input'
import { Label } from './components/ui/label'
import { Badge } from './components/ui/badge'
import { ScrollArea } from './components/ui/scroll-area'
import { Separator } from './components/ui/separator'
import { 
  Search, 
  MessageSquare, 
  FileText, 
  BarChart3, 
  Loader2, 
  BookOpen,
  Scale,
  Gavel,
  Building2
} from 'lucide-react'
import { 
  queryAPI, 
  searchAPI, 
  chatAPI, 
  getStats, 
  type QueryRequest, 
  type ChatMessage,
  type StatsResponse 
} from './lib/utils'

function App() {
  const [activeTab, setActiveTab] = useState('query')
  const [loading, setLoading] = useState(false)
  const [stats, setStats] = useState<StatsResponse | null>(null)
  
  // Query tab state
  const [query, setQuery] = useState('')
  const [queryResult, setQueryResult] = useState<{ answer: string; sources: string[] } | null>(null)
  
  // Search tab state
  const [searchQuery, setSearchQuery] = useState('')
  const [searchResults, setSearchResults] = useState<Array<{ source: string; preview: string }> | null>(null)
  
  // Chat tab state
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([
    { role: 'system', content: 'You are a helpful legal research assistant. How can I help you today?' }
  ])
  const [newMessage, setNewMessage] = useState('')
  
  // Load stats on component mount
  useEffect(() => {
    loadStats()
  }, [])
  
  const loadStats = async () => {
    try {
      const statsData = await getStats()
      setStats(statsData)
    } catch (error) {
      console.error('Failed to load stats:', error)
    }
  }
  
  const handleQuery = async () => {
    if (!query.trim()) return
    
    setLoading(true)
    try {
      const result = await queryAPI({ query: query.trim() })
      setQueryResult(result)
    } catch (error) {
      console.error('Query failed:', error)
      setQueryResult({ answer: 'Sorry, an error occurred while processing your query.', sources: [] })
    } finally {
      setLoading(false)
    }
  }
  
  const handleSearch = async () => {
    if (!searchQuery.trim()) return
    
    setLoading(true)
    try {
      const result = await searchAPI({ query: searchQuery.trim() })
      setSearchResults(result.matches)
    } catch (error) {
      console.error('Search failed:', error)
      setSearchResults([])
    } finally {
      setLoading(false)
    }
  }
  
  const handleChat = async () => {
    if (!newMessage.trim()) return
    
    const userMessage: ChatMessage = { role: 'user', content: newMessage.trim() }
    const updatedMessages = [...chatMessages, userMessage]
    setChatMessages(updatedMessages)
    setNewMessage('')
    
    setLoading(true)
    try {
      const result = await chatAPI({ messages: updatedMessages })
      const assistantMessage: ChatMessage = { role: 'assistant', content: result.answer }
      setChatMessages([...updatedMessages, assistantMessage])
    } catch (error) {
      console.error('Chat failed:', error)
      const errorMessage: ChatMessage = { 
        role: 'assistant', 
        content: 'Sorry, an error occurred while processing your message.' 
      }
      setChatMessages([...updatedMessages, errorMessage])
    } finally {
      setLoading(false)
    }
  }
  
  const clearChat = () => {
    setChatMessages([
      { role: 'system', content: 'You are a helpful legal research assistant. How can I help you today?' }
    ])
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="flex items-center justify-center gap-3 mb-4">
            <Scale className="h-12 w-12 text-blue-600" />
            <h1 className="text-4xl font-bold text-gray-900">Legal Research Assistant</h1>
          </div>
          <p className="text-lg text-gray-600 max-w-2xl mx-auto">
            Powered by AI and comprehensive legal databases. Get instant answers to your legal questions 
            with citations and source references.
          </p>
        </div>
        
        {/* Stats Overview */}
        {stats && (
          <Card className="mb-6">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <BarChart3 className="h-5 w-5" />
                System Status
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <div className="text-center">
                  <div className="text-2xl font-bold text-blue-600">{stats.doc_count.toLocaleString()}</div>
                  <div className="text-sm text-gray-600">Documents</div>
                </div>
                <div className="text-center">
                  <Badge variant={stats.bm25_ready ? "default" : "secondary"}>
                    {stats.bm25_ready ? "Ready" : "Building"}
                  </Badge>
                  <div className="text-sm text-gray-600 mt-1">BM25 Index</div>
                </div>
                <div className="text-center">
                  <div className="text-sm text-gray-600">Collection</div>
                  <div className="font-medium">{stats.collection}</div>
                </div>
                <div className="text-center">
                  <div className="text-sm text-gray-600">Status</div>
                  <Badge variant="default" className="bg-green-100 text-green-800">
                    Active
                  </Badge>
                </div>
              </div>
            </CardContent>
          </Card>
        )}
        
        {/* Main Tabs */}
        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="grid w-full grid-cols-4 mb-6">
            <TabsTrigger value="query" className="flex items-center gap-2">
              <FileText className="h-4 w-4" />
              Query
            </TabsTrigger>
            <TabsTrigger value="search" className="flex items-center gap-2">
              <Search className="h-4 w-4" />
              Search
            </TabsTrigger>
            <TabsTrigger value="chat" className="flex items-center gap-2">
              <MessageSquare className="h-4 w-4" />
              Chat
            </TabsTrigger>
            <TabsTrigger value="stats" className="flex items-center gap-2">
              <BarChart3 className="h-4 w-4" />
              Details
            </TabsTrigger>
          </TabsList>
          
          {/* Query Tab */}
          <TabsContent value="query">
            <Card>
              <CardHeader>
                <CardTitle>Legal Query</CardTitle>
                <CardDescription>
                  Ask a specific legal question and get a comprehensive answer with citations
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex gap-2">
                  <Input
                    placeholder="Enter your legal question..."
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    onKeyPress={(e) => e.key === 'Enter' && handleQuery()}
                    className="flex-1"
                  />
                  <Button onClick={handleQuery} disabled={loading || !query.trim()}>
                    {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Search className="h-4 w-4" />}
                    Ask
                  </Button>
                </div>
                
                {queryResult && (
                  <div className="space-y-4">
                    <Separator />
                    <div>
                      <h3 className="font-semibold mb-2">Answer:</h3>
                      <div className="bg-blue-50 p-4 rounded-lg border-l-4 border-blue-500">
                        {queryResult.answer}
                      </div>
                    </div>
                    
                    {queryResult.sources.length > 0 && (
                      <div>
                        <h3 className="font-semibold mb-2">Sources:</h3>
                        <div className="space-y-2">
                          {queryResult.sources.map((source, index) => (
                            <Badge key={index} variant="outline" className="mr-2">
                              {source.split('/').pop()}
                            </Badge>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>
          
          {/* Search Tab */}
          <TabsContent value="search">
            <Card>
              <CardHeader>
                <CardTitle>Document Search</CardTitle>
                <CardDescription>
                  Search through legal documents and get relevant excerpts
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex gap-2">
                  <Input
                    placeholder="Search for legal terms, cases, or statutes..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
                    className="flex-1"
                  />
                  <Button onClick={handleSearch} disabled={loading || !searchQuery.trim()}>
                    {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Search className="h-4 w-4" />}
                    Search
                  </Button>
                </div>
                
                {searchResults && (
                  <div className="space-y-4">
                    <Separator />
                    <div>
                      <h3 className="font-semibold mb-3">Search Results ({searchResults.length}):</h3>
                      <ScrollArea className="h-96">
                        <div className="space-y-3">
                          {searchResults.map((result, index) => (
                            <Card key={index} className="p-4">
                              <div className="flex items-start gap-3">
                                <BookOpen className="h-5 w-5 text-blue-600 mt-1 flex-shrink-0" />
                                <div className="flex-1">
                                  <div className="font-medium text-sm text-gray-600 mb-2">
                                    Source: {result.source.split('/').pop()}
                                  </div>
                                  <p className="text-gray-800">{result.preview}...</p>
                                </div>
                              </div>
                            </Card>
                          ))}
                        </div>
                      </ScrollArea>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>
          
          {/* Chat Tab */}
          <TabsContent value="chat">
            <Card>
              <CardHeader>
                <CardTitle>Legal Chat Assistant</CardTitle>
                <CardDescription>
                  Have a conversation with your AI legal research assistant
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <ScrollArea className="h-96 border rounded-lg p-4 bg-gray-50">
                  <div className="space-y-4">
                    {chatMessages.map((message, index) => (
                      <div
                        key={index}
                        className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
                      >
                        <div
                          className={`max-w-[80%] p-3 rounded-lg ${
                            message.role === 'user'
                              ? 'bg-blue-600 text-white'
                              : message.role === 'system'
                              ? 'bg-gray-200 text-gray-700'
                              : 'bg-white text-gray-800 border'
                          }`}
                        >
                          <div className="text-xs opacity-70 mb-1">
                            {message.role === 'system' ? 'System' : message.role === 'user' ? 'You' : 'Assistant'}
                          </div>
                          {message.content}
                        </div>
                      </div>
                    ))}
                    {loading && (
                      <div className="flex justify-start">
                        <div className="bg-white text-gray-800 border p-3 rounded-lg">
                          <Loader2 className="h-4 w-4 animate-spin" />
                        </div>
                      </div>
                    )}
                  </div>
                </ScrollArea>
                
                <div className="flex gap-2">
                  <Input
                    placeholder="Type your message..."
                    value={newMessage}
                    onChange={(e) => setNewMessage(e.target.value)}
                    onKeyPress={(e) => e.key === 'Enter' && handleChat()}
                    className="flex-1"
                  />
                  <Button onClick={handleChat} disabled={loading || !newMessage.trim()}>
                    {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <MessageSquare className="h-4 w-4" />}
                    Send
                  </Button>
                  <Button variant="outline" onClick={clearChat}>
                    Clear
                  </Button>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
          
          {/* Stats Tab */}
          <TabsContent value="stats">
            <Card>
              <CardHeader>
                <CardTitle>System Details</CardTitle>
                <CardDescription>
                  Detailed information about the legal database and system configuration
                </CardDescription>
              </CardHeader>
              <CardContent>
                {stats ? (
                  <div className="space-y-6">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      <div>
                        <h3 className="font-semibold mb-3 flex items-center gap-2">
                          <Building2 className="h-4 w-4" />
                          Database Information
                        </h3>
                        <div className="space-y-2 text-sm">
                          <div className="flex justify-between">
                            <span className="text-gray-600">Collection:</span>
                            <span className="font-medium">{stats.collection}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-600">Documents:</span>
                            <span className="font-medium">{stats.doc_count.toLocaleString()}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-600">BM25 Index:</span>
                            <Badge variant={stats.bm25_ready ? "default" : "secondary"}>
                              {stats.bm25_ready ? "Ready" : "Building"}
                            </Badge>
                          </div>
                        </div>
                      </div>
                      
      <div>
                        <h3 className="font-semibold mb-3 flex items-center gap-2">
                          <Gavel className="h-4 w-4" />
                          Data Sources
                        </h3>
                        <div className="space-y-2 text-sm">
                          <div className="flex justify-between">
                            <span className="text-gray-600">Cases Directory:</span>
                            <span className="font-medium">{stats.cases_dir.split('/').pop()}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-600">Statutes Directory:</span>
                            <span className="font-medium">{stats.statutes_dir.split('/').pop()}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-600">Corpus File:</span>
                            <span className="font-medium">{stats.corpus_file.split('/').pop()}</span>
                          </div>
                        </div>
                      </div>
                    </div>
                    
                    <Separator />
                    
                    <div className="text-center">
                      <Button onClick={loadStats} variant="outline">
                        Refresh Stats
                      </Button>
                    </div>
                  </div>
                ) : (
                  <div className="text-center py-8">
                    <Loader2 className="h-8 w-8 animate-spin mx-auto mb-4" />
                    <p className="text-gray-600">Loading system information...</p>
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
      </div>
  )
}

export default App
