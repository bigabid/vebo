import { Book, Loader2 } from 'lucide-react';
import { Card } from '@/components/ui/card';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { useCatalogs } from '@/hooks/useAthenaApi';

interface CatalogSelectorProps {
  dataSource: string | null;
  selectedCatalog: string | null;
  onChange: (catalog: string) => void;
}

export function CatalogSelector({ dataSource, selectedCatalog, onChange }: CatalogSelectorProps) {
  const { data, isLoading, error } = useCatalogs(dataSource);

  if (!dataSource) return null;

  return (
    <Card className="p-6 bg-gradient-card border-0 shadow-soft">
      <div className="flex items-center gap-3 mb-4">
        <div className="p-2 bg-primary/10 rounded-lg">
          <Book className="w-5 h-5 text-primary" />
        </div>
        <div>
          <h3 className="font-semibold text-foreground">Select Catalog</h3>
          <p className="text-sm text-muted-foreground">Choose a Glue/Athena catalog</p>
        </div>
      </div>

      {error ? (
        <div className="p-4 bg-destructive/10 border border-destructive/20 rounded-lg">
          <p className="text-sm text-destructive font-medium">Failed to load catalogs</p>
        </div>
      ) : (
        <Select value={selectedCatalog || ''} onValueChange={onChange} disabled={isLoading}>
          <SelectTrigger className="w-full h-12 bg-background border-2 border-border hover:border-primary/50 transition-colors">
            <SelectValue
              placeholder={
                isLoading ? (
                  <div className="flex items-center gap-2">
                    <Loader2 className="w-4 h-4 animate-spin text-muted-foreground" />
                    <span>Loading catalogs...</span>
                  </div>
                ) : (
                  'Select a catalog'
                )
              }
            />
          </SelectTrigger>
          <SelectContent className="bg-card border-2 border-border shadow-medium">
            {data?.catalogs.map((name) => (
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


