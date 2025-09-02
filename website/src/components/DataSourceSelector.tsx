import { Server, Loader2 } from 'lucide-react';
import { Card } from '@/components/ui/card';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { useDataSources } from '@/hooks/useAthenaApi';

interface DataSourceSelectorProps {
  selectedDataSource: string | null;
  onChange: (dataSource: string) => void;
}

export function DataSourceSelector({ selectedDataSource, onChange }: DataSourceSelectorProps) {
  const { data, isLoading, error } = useDataSources();

  return (
    <Card className="p-6 bg-gradient-card border-0 shadow-soft">
      <div className="flex items-center gap-3 mb-4">
        <div className="p-2 bg-primary/10 rounded-lg">
          <Server className="w-5 h-5 text-primary" />
        </div>
        <div>
          <h3 className="font-semibold text-foreground">Select Data Source</h3>
          <p className="text-sm text-muted-foreground">Choose an Athena data source</p>
        </div>
      </div>

      {error ? (
        <div className="p-4 bg-destructive/10 border border-destructive/20 rounded-lg">
          <p className="text-sm text-destructive font-medium">Failed to load data sources</p>
        </div>
      ) : (
        <Select value={selectedDataSource || ''} onValueChange={onChange} disabled={isLoading}>
          <SelectTrigger className="w-full h-12 bg-background border-2 border-border hover:border-primary/50 transition-colors">
            <SelectValue
              placeholder={
                isLoading ? (
                  <div className="flex items-center gap-2">
                    <Loader2 className="w-4 h-4 animate-spin text-muted-foreground" />
                    <span>Loading data sources...</span>
                  </div>
                ) : (
                  'Select a data source'
                )
              }
            />
          </SelectTrigger>
          <SelectContent className="bg-card border-2 border-border shadow-medium">
            {data?.dataSources.map((name) => (
              <SelectItem key={name} value={name} className="hover:bg-accent cursor-pointer py-3 px-4">
                {name}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      )}
    </Card>
  );
}


