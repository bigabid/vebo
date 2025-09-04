import React, { useState, useEffect } from 'react';
import { Code, Play, RefreshCw, AlertTriangle, Copy, Check } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { useToast } from '@/hooks/use-toast';
import { Textarea } from '@/components/ui/textarea';
import { useGenerateQuery } from '@/hooks/useAthenaApi';

interface QueryEditorProps {
  dataSource: string | null;
  catalog: string | null;
  database: string | null;
  table: string | null;
  partitions: Record<string, string[]>;
  onQueryChange?: (query: string) => void;
}

export function QueryEditor({
  dataSource,
  catalog,
  database,
  table,
  partitions,
  onQueryChange
}: QueryEditorProps) {
  const [query, setQuery] = useState('');
  const [originalQuery, setOriginalQuery] = useState('');
  const [explanation, setExplanation] = useState('');
  const [partitionsApplied, setPartitionsApplied] = useState<string[]>([]);
  const [isModified, setIsModified] = useState(false);
  const [copied, setCopied] = useState(false);
  const { toast } = useToast();
  
  const generateQueryMutation = useGenerateQuery();

  // Generate query when table selection changes
  useEffect(() => {
    if (catalog && database && table) {
      generateQuery();
    } else {
      setQuery('');
      setOriginalQuery('');
      setExplanation('');
      setPartitionsApplied([]);
      setIsModified(false);
    }
  }, [catalog, database, table, partitions]);

  const generateQuery = async () => {
    if (!catalog || !database || !table) return;

    try {
      const data = await generateQueryMutation.mutateAsync({
        catalog,
        database,
        table,
        partitions
      });
      
      setQuery(data.query);
      setOriginalQuery(data.query);
      setExplanation(data.explanation);
      setPartitionsApplied(data.partitionsApplied || []);
      setIsModified(false);
    } catch (error) {
      toast({
        title: "Failed to generate query",
        description: error instanceof Error ? error.message : "Unknown error",
        variant: "destructive",
      });
    }
  };

  const handleQueryChange = (value: string) => {
    setQuery(value);
    setIsModified(value !== originalQuery);
    if (onQueryChange) {
      onQueryChange(value);
    }
  };

  // Always notify parent of current query, including on initial generation
  useEffect(() => {
    if (onQueryChange && query) {
      onQueryChange(query);
    }
  }, [query, onQueryChange]);

  const handleReset = () => {
    setQuery(originalQuery);
    setIsModified(false);
  };

  const copyToClipboard = async () => {
    try {
      await navigator.clipboard.writeText(query);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
      toast({
        title: "Copied to clipboard",
        description: "Query copied successfully",
      });
    } catch (err) {
      toast({
        title: "Failed to copy",
        description: "Could not copy query to clipboard",
        variant: "destructive",
      });
    }
  };

  const hasTable = catalog && database && table;

  return (
    <Card className="p-6">
      <div className="space-y-4">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Code className="w-5 h-5" />
            <h3 className="text-lg font-semibold">SQL Query</h3>
            {isModified && (
              <Badge variant="secondary" className="text-xs">
                Modified
              </Badge>
            )}
          </div>
          <div className="flex items-center gap-2">
            {hasTable && (
              <>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={generateQuery}
                  disabled={generateQueryMutation.isPending}
                  className="gap-2"
                >
                  <RefreshCw className={`w-4 h-4 ${generateQueryMutation.isPending ? 'animate-spin' : ''}`} />
                  Regenerate
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={copyToClipboard}
                  className="gap-2"
                >
                  {copied ? <Check className="w-4 h-4" /> : <Copy className="w-4 h-4" />}
                  {copied ? 'Copied' : 'Copy'}
                </Button>
              </>
            )}
          </div>
        </div>

        {/* Query Info */}
        {explanation && (
          <div className="text-sm text-muted-foreground">
            {explanation}
            {partitionsApplied.length > 0 && (
              <div className="mt-1">
                <span className="font-medium">Partitions applied:</span> {partitionsApplied.join(', ')}
              </div>
            )}
          </div>
        )}

        {/* Query Editor */}
        {hasTable ? (
          <>
            {generateQueryMutation.isPending ? (
              <div className="flex items-center justify-center py-12 text-muted-foreground">
                <RefreshCw className="w-6 h-6 animate-spin mr-2" />
                Generating query...
              </div>
            ) : (
              <div className="space-y-3">
                <Textarea
                  value={query}
                  onChange={(e) => handleQueryChange(e.target.value)}
                  placeholder="SQL query will be generated here..."
                  className="font-mono text-sm min-h-[200px] resize-y"
                  style={{
                    fontFamily: 'ui-monospace, SFMono-Regular, "SF Mono", Monaco, Consolas, "Liberation Mono", "Courier New", monospace',
                  }}
                />
                
                {/* Action Buttons */}
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    {isModified && (
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={handleReset}
                        className="text-muted-foreground"
                      >
                        Reset to original
                      </Button>
                    )}
                  </div>
                </div>
              </div>
            )}
          </>
        ) : (
          <div className="flex flex-col items-center justify-center py-12 text-muted-foreground space-y-2">
            <AlertTriangle className="w-8 h-8" />
            <p className="text-center">
              Select a catalog, database, and table to generate a SQL query
            </p>
          </div>
        )}

        {/* Query Tips */}
        {hasTable && !generateQueryMutation.isPending && (
          <div className="bg-muted/50 rounded-lg p-4 text-sm">
            <h4 className="font-medium mb-2">Query Tips:</h4>
            <ul className="text-muted-foreground space-y-1 text-xs">
              <li>• You can modify the SELECT clause to include only specific columns</li>
              <li>• Add WHERE conditions to filter the data before profiling</li>
              <li>• Adjust the LIMIT to control sample size (default: 10,000 rows)</li>
              <li>• Use proper SQL syntax for Athena/Presto</li>
            </ul>
          </div>
        )}
      </div>
    </Card>
  );
}
