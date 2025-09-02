import { Database, ChevronDown, Loader2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { useTables } from '@/hooks/useAthenaApi';
import { Card } from '@/components/ui/card';

interface TableSelectorProps {
  selectedTable: string | null;
  onTableSelect: (table: string) => void;
}

export function TableSelector({ selectedTable, onTableSelect }: TableSelectorProps) {
  const { data, isLoading, error } = useTables();

  return (
    <Card className="p-6 bg-gradient-card border-0 shadow-soft">
      <div className="flex items-center gap-3 mb-4">
        <div className="p-2 bg-primary/10 rounded-lg">
          <Database className="w-5 h-5 text-primary" />
        </div>
        <div>
          <h3 className="font-semibold text-foreground">Select Table</h3>
          <p className="text-sm text-muted-foreground">Choose an Athena table to analyze</p>
        </div>
      </div>

      {error ? (
        <div className="p-4 bg-destructive/10 border border-destructive/20 rounded-lg">
          <p className="text-sm text-destructive font-medium">Failed to load tables</p>
          <p className="text-xs text-destructive/80 mt-1">Please check your connection and try again</p>
        </div>
      ) : (
        <Select value={selectedTable || ''} onValueChange={onTableSelect} disabled={isLoading}>
          <SelectTrigger className="w-full h-12 bg-background border-2 border-border hover:border-primary/50 transition-colors">
            <SelectValue placeholder={
              isLoading ? (
                <div className="flex items-center gap-2">
                  <Loader2 className="w-4 h-4 animate-spin text-muted-foreground" />
                  <span>Loading tables...</span>
                </div>
              ) : (
                "Select a table to analyze"
              )
            } />
          </SelectTrigger>
          <SelectContent className="bg-card border-2 border-border shadow-medium">
            {data?.tables.map((table) => (
              <SelectItem 
                key={table.name} 
                value={table.name}
                className="hover:bg-accent cursor-pointer py-3 px-4"
              >
                <div className="flex items-center gap-3">
                  <Database className="w-4 h-4 text-primary" />
                  <span className="font-medium">{table.name}</span>
                </div>
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      )}
    </Card>
  );
}