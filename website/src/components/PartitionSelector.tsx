import { useMemo } from 'react';
import { Layers, CheckSquare, Square, Loader2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { SearchableMultiSelect } from '@/components/ui/searchable-select';
import { usePartitions } from '@/hooks/useAthenaApi';

interface PartitionSelectorProps {
  catalog?: string | null;
  database?: string | null;
  selectedTable: string | null;
  selectedPartitions: Record<string, string[]>;
  onPartitionChange: (partitions: Record<string, string[]>) => void;
}

export function PartitionSelector({ 
  catalog,
  database,
  selectedTable, 
  selectedPartitions, 
  onPartitionChange 
}: PartitionSelectorProps) {
  const { data, isLoading, error } = usePartitions(selectedTable, { catalog: catalog || undefined, database: database || undefined });

  // Group partitions by keys for better organization
  const groupedPartitions = useMemo(() => {
    if (!data?.partitions.length) return {};
    
    const groups: Record<string, Set<string>> = {};
    data.partitionKeys.forEach(key => {
      groups[key] = new Set();
    });
    
    data.partitions.forEach(partition => {
      data.partitionKeys.forEach(key => {
        if (partition[key]) {
          groups[key].add(partition[key]);
        }
      });
    });
    
    return Object.fromEntries(
      Object.entries(groups).map(([key, values]) => [key, Array.from(values).sort()])
    );
  }, [data]);

  const handlePartitionToggle = (key: string, value: string, checked: boolean) => {
    const currentValues = selectedPartitions[key] || [];
    const newValues = checked 
      ? [...currentValues, value]
      : currentValues.filter(v => v !== value);
    
    onPartitionChange({
      ...selectedPartitions,
      [key]: newValues
    });
  };

  const handleSelectAll = (key: string) => {
    const allValues = groupedPartitions[key] || [];
    onPartitionChange({
      ...selectedPartitions,
      [key]: allValues
    });
  };

  const handleClearAll = (key: string) => {
    onPartitionChange({
      ...selectedPartitions,
      [key]: []
    });
  };

  const totalSelected = Object.values(selectedPartitions).flat().length;

  if (!selectedTable) {
    return null;
  }

  return (
    <Card className="p-6 bg-gradient-card border-0 shadow-soft">
      <div className="flex items-center gap-3 mb-4">
        <div className="p-2 bg-secondary/10 rounded-lg">
          <Layers className="w-5 h-5 text-secondary" />
        </div>
        <div className="flex-1">
          <h3 className="font-semibold text-foreground">Filter by Partitions</h3>
          <p className="text-sm text-muted-foreground">
            {selectedTable} - {totalSelected > 0 && <Badge variant="secondary" className="ml-1">{totalSelected} selected</Badge>}
          </p>
        </div>
      </div>

      {isLoading ? (
        <div className="flex items-center justify-center py-8">
          <Loader2 className="w-6 h-6 animate-spin text-primary" />
          <span className="ml-2 text-muted-foreground">Loading partitions...</span>
        </div>
      ) : error ? (
        <div className="p-4 bg-destructive/10 border border-destructive/20 rounded-lg">
          <p className="text-sm text-destructive font-medium">Failed to load partitions</p>
        </div>
      ) : !data?.partitionKeys.length ? (
        <div className="p-4 bg-muted/50 rounded-lg border border-border">
          <p className="text-sm text-muted-foreground text-center">
            This table has no partitions
          </p>
        </div>
      ) : (
        <div className="space-y-6">
          {data.partitionKeys.map((key) => {
            const values = groupedPartitions[key] || [];
            const selectedValues = selectedPartitions[key] || [];
            const allSelected = selectedValues.length === values.length;
            const someSelected = selectedValues.length > 0;

            return (
              <div key={key} className="space-y-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <h4 className="font-medium text-foreground capitalize">{key}</h4>
                    <Badge variant="outline" className="text-xs">
                      {selectedValues.length} / {values.length}
                    </Badge>
                  </div>
                  <div className="flex gap-2">
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => handleSelectAll(key)}
                      disabled={allSelected}
                      className="h-7 px-2 text-xs"
                    >
                      <CheckSquare className="w-3 h-3 mr-1" />
                      Select All
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => handleClearAll(key)}
                      disabled={!someSelected}
                      className="h-7 px-2 text-xs"
                    >
                      <Square className="w-3 h-3 mr-1" />
                      Clear
                    </Button>
                  </div>
                </div>
                
                <SearchableMultiSelect
                  values={selectedValues}
                  onChange={(vals) => {
                    onPartitionChange({
                      ...selectedPartitions,
                      [key]: vals,
                    })
                  }}
                  options={values}
                  placeholder={values.length ? `Select ${key}` : `No ${key} values`}
                />
              </div>
            );
          })}
        </div>
      )}
    </Card>
  );
}